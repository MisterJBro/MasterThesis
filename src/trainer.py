import numpy as np
import torch
import torch.nn as nn

import gym
import time
import random
from src.envs import Envs
from src.model import ValueEquivalenceModel
from src.plan import plan
from src.policy import ActorCriticPolicy

from multiprocessing import freeze_support
from src.process import post_processing
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from tabulate import tabulate
from src.sample_batch import SampleBatch
from src.model import to_onehot
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions.kl import kl_divergence


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
            start = time.time()
            sample_batch = self.get_sample_batch()
            end = time.time()
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
        ret = data["ret"]
        val = data["val"]

        # Time
        start = time.time()
        plan_targets = plan(self.policy, self.model, data, self.config)
        plan_actions = plan_targets.logits.argmax(-1)
        end = time.time()
        #print(f'Plan time: {end - start}')

        # Distill planning targets into policy
        trainset = TensorDataset(obs, plan_targets.logits)
        trainloader = DataLoader(trainset, batch_size=int(self.config["num_samples"]/10), shuffle=True)

        for i in range(20):
            for obs_batch, plan_target_batch in trainloader:
                self.policy.opt_policy.zero_grad()
                dist_batch = self.policy.get_dist(obs_batch)
                loss = kl_divergence(Categorical(logits=plan_target_batch), dist_batch).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.policy.parameters(),  self.config["grad_clip"])
                self.policy.opt_policy.step()

        #act = data["act"]
        #act_onehot = to_onehot(act, self.config["num_acts"])
        #with torch.no_grad():
        #    act_model = self.model.dyn_linear(act_onehot)
        #    state = self.model.representation(obs)
        #    hidden, _ = self.model.dynamics(state, act_model)
        #    model_val = self.model.get_value(hidden)
        #    model_rew = self.model.get_reward(hidden)
        #q_val = model_rew + self.config["gamma"] * model_val

        adv = ret - val
        data["adv"] = adv

        # Policy and Value loss
        #self.policy.loss_gradient(data)
        self.policy.loss_value(data)

        # Get new logits for model loss
        with torch.no_grad():
            data["logits"] = self.policy.get_dist(obs).logits

        # Model loss
        self.model.loss(data)

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
