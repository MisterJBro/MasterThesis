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
        # Get data
        obs = torch.from_numpy(sample_batch.obs).float()
        obs = obs.reshape(-1, obs.shape[-1]).to(self.device)
        act = torch.from_numpy(sample_batch.act).long().reshape(-1).to(self.device)
        rew = torch.from_numpy(sample_batch.rew).float().reshape(-1).to(self.device)
        ret = torch.from_numpy(sample_batch.ret).float().reshape(-1).to(self.device)
        val = torch.from_numpy(sample_batch.val).float().reshape(-1).to(self.device)

        adv = ret - val
        scalar_loss = nn.HuberLoss()

        # Policy loss
        self.policy.opt_policy.zero_grad()

        dist = self.policy.get_dist(obs)
        logp = dist.log_prob(act)
        loss_policy = -(logp * adv).mean()
        loss_policy.backward()
        nn.utils.clip_grad_norm_(self.policy.policy.parameters(),  self.config["grad_clip"])
        self.policy.opt_policy.step()

        # Value loss
        trainset = torch.utils.data.TensorDataset(obs, ret)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=int(self.config["num_samples"]/5), shuffle=True)

        for _ in range(self.config["vf_iters"]):
            for obs_batch, ret_batch in trainloader:
                self.policy.opt_value.zero_grad()
                val_batch = self.policy.get_value(obs_batch)
                loss_value = scalar_loss(val_batch, ret_batch)
                loss_value.backward()
                nn.utils.clip_grad_norm_(self.policy.value.parameters(),  self.config["grad_clip"])
                self.policy.opt_value.step()

        # Model loss
        with torch.no_grad():
            logits_policy = self.policy.get_dist(obs).logits
        act_onehot = to_onehot(act, self.config["num_acts"])
        sections = sample_batch.get_sections()

        # Create episode information
        episodes = []
        for (start, end) in sections:
            lengths = []
            act_ep = []
            rew_targets = []
            ret_targets = []
            dist_targets = []
            for t in range(start, end):
                end_ep = min(t+self.config["model_unroll_len"], end)
                next_actions = act_onehot[t:end_ep].squeeze(1)
                next_rews = rew[t:end_ep]
                next_rets = ret[t:end_ep]
                next_logits = logits_policy[t:end_ep]

                rew_targets.append(next_rews)
                ret_targets.append(next_rets)
                dist_targets.append(next_logits)
                act_ep.append(next_actions)
                lengths.append(next_actions.shape[0])
            rew_targets = torch.concat(rew_targets)
            ret_targets = torch.concat(ret_targets)
            dist_targets = Categorical(logits=torch.concat(dist_targets))
            act_ep = pad_sequence(act_ep, batch_first=True)
            act_ep = pack_padded_sequence(act_ep, lengths=lengths, batch_first=True)
            episodes.append({
                'rew_targets': rew_targets,
                'ret_targets': ret_targets,
                'dist_targets': dist_targets,
                'act_ep': act_ep,
                'start': start,
                'end': end,
            })

        self.config["model_iters"] = 3
        import time
        start_time = time.time()
        for _ in range(self.config["model_iters"]):
            self.model.opt.zero_grad()
            (h_0, c_0) = self.model.representation(obs)

            # Iterate over episodes
            losses = []
            for ep in episodes:
                start = ep["start"]
                end = ep["end"]
                act_ep = ep["act_ep"]
                rew_targets = ep["rew_targets"]
                ret_targets = ep["ret_targets"]
                dist_targets = ep["dist_targets"]
                s_ep = (h_0[:, start:end].contiguous(), c_0[:, start:end].contiguous())

                # Calculate dynamics by unrolling the model
                hidden, _ = self.model.dynamics(s_ep, act_ep)
                hidden, lengths = pad_packed_sequence(hidden, batch_first=True)
                hidden = [hidden[i][:len] for i, len in enumerate(lengths)]
                hidden = torch.concat(hidden)

                # Calculate rew, val and pi distributions and losses
                pred_rew = self.model.get_reward(hidden)
                pred_val = self.model.get_value(hidden)
                dist_model = self.model.get_policy(hidden)

                loss_rew = scalar_loss(pred_rew, rew_targets)
                loss_val = scalar_loss(pred_val, ret_targets)
                loss_pi = kl_divergence(dist_model, dist_targets).mean()
                loss = 0.5 * loss_rew + 0.01 * loss_val + loss_pi
                losses.append(loss)
            loss = torch.stack(losses).mean()
            loss.backward()
            print("Model loss: ", loss.item())

            nn.utils.clip_grad_norm_(self.model.parameters(),  self.config["grad_clip"])
            self.model.opt.step()
        print("Model update time: ", time.time() - start_time)


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

def to_tensors(list, device):
    return [torch.as_tensor(elem).to(device) for elem in list]

def to_onehot(a, num_acts):
    # Convert action into one-hot encoded representation
    a_onehot = torch.zeros(a.shape[0], num_acts).to(a.device)
    a_onehot.scatter_(1, a.view(-1, 1), 1)
    a_onehot.unsqueeze_(1)

    return a_onehot