from math import dist
import numpy as np
import torch
import torch.nn as nn

import gym
import time
import random
from src.env.envs import Envs
from src.networks.model import ValueEquivalenceModel
from src.networks.policy_pend import PendulumPolicy
from src.search.alpha_zero import AlphaZero
from src.search.model_search import plan
from src.search.state import State

from multiprocessing import freeze_support
from src.train.processer import post_processing
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from tabulate import tabulate
from src.env.sample_batch import SampleBatch
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions.kl import kl_divergence


class AZExitTrainer:
    def __init__(self, config):
        # RNG seed
        seed = config["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.config = config
        self.device = config["device"]
        self.num_acts = config["num_acts"]
        self.envs = Envs(config)
        self.policy = PendulumPolicy(config)
        self.az = AlphaZero(self.policy, config)
        self.writer = SummaryWriter(log_dir="../runs",comment=f'{config["env"]}_{config["num_samples"]}')
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
            self.update(sample_batch)
            stats = sample_batch.statistics

            avg_ret = stats["mean_return"]
            max_ret = stats["max_return"]
            min_ret = stats["min_return"]
            print(f'Iteration: {iter}  Avg Ret: {np.round(avg_ret, 3)}  Max Ret: {np.round(max_ret, 3)}  Min Ret: {np.round(min_ret, 3)}')
            self.writer.add_scalar('Average return', avg_ret, iter)

    def get_sample_batch(self):
        self.az.update_policy(self.policy.state_dict())
        sample_batch = SampleBatch(self.config)
        obs = self.envs.reset()

        for _ in range(self.config["sample_len"]):
            import time
            start = time.time()
            act, dist = self.search_action(obs)
            obs_next, rew, done = self.envs.step(act)
            print(f'Time: {time.time() - start}')

            sample_batch.append(obs, act, rew, done, dist=dist)
            obs = obs_next

        sample_batch.set_last_obs(obs)
        sample_batch = post_processing(self.policy, sample_batch, self.config)
        return sample_batch

    def search_action(self, obs):
        envs = self.envs.get_all_env()
        states = [State(env, obs=obs[i]) for i, env in enumerate(envs)]

        dist = self.az.distributed_search(states)
        act = np.concatenate([np.random.choice(self.num_acts, 1, p=p) for p in dist])

        return act, dist

    def update(self, sample_batch):
        data = sample_batch.to_tensor_dict()
        obs = data["obs"]
        dist = data["dist"]
        ret = data["ret"]
        scalar_loss = nn.HuberLoss()

        # Distill planning targets into policy
        trainset = TensorDataset(obs, dist, ret)
        trainloader = DataLoader(trainset, batch_size=int(self.config["num_samples"]/10), shuffle=True)

        for i in range(10):
            for obs_batch, target_batch, ret_batch in trainloader:
                self.policy.opt.zero_grad()

                dist_batch, val_batch = self.policy(obs_batch)
                loss_dist = kl_divergence(Categorical(logits=target_batch), dist_batch).mean()
                loss_value = scalar_loss(val_batch, ret_batch)
                loss = loss_dist + loss_value
                loss.backward()

                nn.utils.clip_grad_norm_(self.policy.parameters(),  self.config["grad_clip"])
                self.policy.opt.step()

    def test(self):
        if isinstance(self.config["env"], str):
            env = gym.make(self.config["env"])
        else:
            env = self.config["env"]
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


