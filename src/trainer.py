import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import gym
import time
import random
from src.envs import Envs
from src.config import DEFAULT_CONFIG
from src.model import ValueEquivalenceModel
from src.policy import ActorCriticPolicy

from multiprocessing import freeze_support
from src.process import post_processing
from torch.utils.tensorboard import SummaryWriter
from tabulate import tabulate
from src.sample_batch import SampleBatch


class Trainer:
    def __init__(self, config):
        # RNG seed
        seed = config["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.config = config
        self.device = config["device"]
        self.envs = Envs(config)
        self.policy = ActorCriticPolicy(config)
        self.model = ValueEquivalenceModel(config)
        self.writer = SummaryWriter(comment=f'{config["env"]}_{config["num_samples"]}')
        self.max_avg_rew = float('-inf')

        print(tabulate([
            ['Environment', config["env"]],
            ['Obs shape', config["obs_dim"]],
            ['Actions num', config["num_acts"]],
            ['CPU count', config["num_cpus"]],
        ], colalign=("left", "right")))
        print()

    def train(self):
        for iter in range(self.config["train_iters"]):
            sample_batch = self.get_sample_batch()
            stats = sample_batch.statistics
            self.update(sample_batch)

            avg_ret = stats["mean_return"]
            max_ret = stats["max_return"]
            min_ret = stats["min_return"]
            print(f'Iteration: {iter}  Avg Ret: {np.round(avg_ret, 3)}  Max Ret: {np.round(max_ret, 3)}  Min Ret: {np.round(min_ret, 3)}')
            self.writer.add_scalar('Average return', avg_ret, iter)

    def get_sample_batch(self):
        sample_batch = SampleBatch(self.config["num_envs"], self.config)
        obs = self.envs.reset()

        for _ in range(self.config["sample_len"]):
            act = self.policy.get_action(obs)
            obs_next, rew, done = self.envs.step(act)

            sample_batch.append(obs, act, rew, done)
            obs = obs_next

        sample_batch.set_last_obs(obs)
        sample_batch = post_processing(self.policy, sample_batch, self.config)
        return sample_batch

    def update(self, sample_batch):
        data = sample_batch.to_tensor_dict()
        obs = data["obs"]
        act = data["act"]
        rew = data["rew"]
        done = data["done"]
        ret = data["ret"]
        val = data["val"]
        last_val = data["last_val"]
        scalar_loss = nn.HuberLoss()
        sections = sample_batch.get_sections()
        act_onehot = to_onehot(act, self.config["num_acts"])

        with torch.no_grad():
            act_model = self.model.dyn_linear(act_onehot)
            state = self.model.representation(obs)
            hidden, _ = self.model.dynamics(state, act_model)
            model_val = self.model.get_value(hidden)
            model_rew = self.model.get_reward(hidden)
        q_val = model_rew + self.config["gamma"] * model_val

        adv = q_val - val
        data["adv"] = adv

        # Policy and Value loss
        self.policy.loss(data)

        # Model loss
        with torch.no_grad():
            logits_policy = self.policy.get_dist(obs).logits

        episodes = []
        num_bootstraps = 0
        for (start, end) in sections:
            act_ep = []
            lengths = []
            rew_targets = []
            val_targets = []
            dist_targets = []
            if done[end-1]:
                next_val_targets = torch.concat([ret[start+1:end], torch.zeros(1, device=self.device)])
            else:
                next_val_targets = torch.concat([ret[start+1:end], last_val[num_bootstraps].unsqueeze(0)])
                num_bootstraps += 1

            for t in range(start, end):
                end_ep = min(t+self.config["model_unroll_len"], end)
                next_actions = act_onehot[t:end_ep].squeeze(1)
                rew_target = rew[t:end_ep]
                val_target = next_val_targets[t-start:end_ep-start]
                dist_target = logits_policy[t:end_ep]

                rew_targets.append(rew_target)
                val_targets.append(val_target)
                dist_targets.append(dist_target)
                act_ep.append(next_actions)
                lengths.append(next_actions.shape[0])

            act_ep = torch.concat(act_ep)
            rew_targets = torch.concat(rew_targets)
            val_targets = torch.concat(val_targets)
            dist_targets = torch.concat(dist_targets)
            dist_targets = Categorical(logits=dist_targets)

            episodes.append({
                'start': start,
                'end': end,
                'act_ep': act_ep,
                'lengths': lengths,
                'rew_targets': rew_targets,
                'val_targets': val_targets,
                'dist_targets': dist_targets,
            })
        batch_size = int(self.config["num_samples"]/self.config["model_minibatches"])

        for _ in range(self.config["model_iters"]):
            losses = []
            steps = 0
            np.random.shuffle(episodes)
            for ep in episodes:
                start = ep["start"]
                end = ep["end"]
                act_ep = ep["act_ep"]
                lengths = ep["lengths"]
                rew_targets = ep["rew_targets"]
                val_targets = ep["val_targets"]
                dist_targets = ep["dist_targets"]

                act_ep = self.model.dyn_linear(act_ep)
                tmp = []
                tmp_index = 0
                for l in lengths:
                    tmp.append(act_ep[tmp_index:tmp_index + l])
                    tmp_index += l
                act_ep = tmp
                act_ep = pad_sequence(act_ep, batch_first=True)
                act_ep = pack_padded_sequence(act_ep, lengths=lengths, batch_first=True)

                state = self.model.representation(obs[start:end])
                hidden, _ = self.model.dynamics(state, act_ep)
                hidden, lengths = pad_packed_sequence(hidden, batch_first=True)
                hidden = [hidden[i][:len] for i, len in enumerate(lengths)]
                hidden = torch.concat(hidden)

                model_rew = self.model.get_reward(hidden)
                model_val = self.model.get_value(hidden)
                model_dist = self.model.get_policy(hidden)

                loss_rew = scalar_loss(model_rew, rew_targets)
                loss_val = scalar_loss(model_val, val_targets)
                # Mode seeking KL
                loss_dist = kl_divergence(dist_targets, model_dist).mean()
                loss = loss_rew + loss_val + loss_dist
                losses.append(loss)

                # Minibatch update
                steps += end - start
                if steps >= batch_size:
                    self.model.opt.zero_grad()
                    loss = torch.mean(torch.stack(losses))
                    loss.backward()
                    print(f"Steps {steps} Model loss: {loss.item()}  Loss Reward: {np.round(loss_rew.item(),3)}  Loss Value: {np.round(loss_val.item(), 3)}  Loss KL: {np.round(loss_dist.item(), 3)}")
                    self.model.opt.step()
                    steps = 0
                    losses = []

            if len(losses) != 0:
                self.model.opt.zero_grad()
                loss = torch.mean(torch.stack(losses))
                loss.backward()
                print(f"Steps {steps} Final Model loss: {loss.item()}")
                self.model.opt.step()

    def test(self):
        env = gym.make(self.config["env"])
        rews = []
        input('Press any key to continue...')

        obs = env.reset()
        for _ in range(self.config["test_len"]):
            env.render()
            act = self.policy.get_action(obs)
            obs, rew, done, _ = env.step(act)
            rews.append(rew)
            if done:
                break
        print(f'Undiscounted return: {np.sum(rews)}')
        env.close()

    def __enter__(self):
        freeze_support()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.envs.close()
        self.writer.flush()
        self.writer.close()

def to_onehot(a, num_acts):
    # Convert action into one-hot encoded representation
    a_onehot = torch.zeros(a.shape[0], num_acts).to(a.device)
    a_onehot.scatter_(1, a.view(-1, 1), 1)
    a_onehot.unsqueeze_(1)

    return a_onehot