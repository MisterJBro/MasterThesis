import numpy as np
import torch
import torch.nn as nn
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
        criterion = nn.MSELoss()

        # Loss
        self.policy.opt_policy.zero_grad()

        dist = self.policy.get_dist(obs)
        logp = dist.log_prob(act)
        loss_policy = -(logp * adv).mean()
        loss_policy.backward()
        #nn.utils.clip_grad_norm_(self.policy.parameters(), 100.0)
        self.policy.opt_policy.step()

        # Critic learn
        self.config["vf_iters"] = 3
        trainset = torch.utils.data.TensorDataset(obs, ret)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=int(self.config["num_samples"]/10), shuffle=True)

        for i in range(self.config["vf_iters"]):
            for obs_batch, ret_batch in trainloader:
                self.policy.opt_value.zero_grad()
                val_batch = self.policy.get_value(obs_batch)
                loss_value = criterion(val_batch, ret_batch)
                loss_value.backward()
                self.policy.opt_value.step()

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
        self.writer.flush()
        self.writer.close()
        self.envs.close()
