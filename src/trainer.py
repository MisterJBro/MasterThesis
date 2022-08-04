import numpy as np
import torch
import gym
import time
import random
from src.envs import Envs
from src.config import DEFAULT_CONFIG
from src.model import ValueEquivalenceModel
from src.policy import ActorCriticPolicy

from .other.utils import to_tensors
import multiprocessing as mp
from multiprocessing import freeze_support
from src.process import post_processing
from torch.utils.tensorboard import SummaryWriter
from tabulate import tabulate


class Trainer:
    def __init__(self, config):
        # RNG seed
        seed = config["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.config = config
        self.envs = Envs(config)
        self.policy = ActorCriticPolicy(config)
        self.model = ValueEquivalenceModel(config)
        self.writer = SummaryWriter('runs/ppo_cartpole')
        self.max_avg_rew = float('-inf')

        print(tabulate([
            ['Environment', config["env"]],
            ['Obs shape', config["obs_dim"]],
            ['Actions num', config["num_acts"]],
            ['CPU count', config["num_cpus"]],
        ], tablefmt="github", colalign=("left", "right")))

    def train(self):
        for epoch in range(self.config["train_iters"]):
            sample_batch = self.get_sample_batch()
            if avg_rew > self.max_avg_rew:
                self.max_avg_rew = avg_rew
                self.net.save()
            self.update()

            print('Epoch {:3}  Avg Rew: {:3}'.format(epoch, avg_rew))
            self.writer.add_scalar('Average reward', avg_rew, epoch)
        self.writer.flush()
        self.writer.close()
        self.envs.close()

    def get_sample_batch(self):
        params = self.policy.state_dict()
        sample_batch = self.envs.sample_batch(params)
        sample_batch = post_processing(self.policy, sample_batch, config)
        return sample_batch

    def compute_policy_gradient(self, dist, act, adv, old_logp, clip_ratio=0.2):
        logp = dist.log_prob(act)

        ratio = torch.exp(logp - old_logp)
        clipped = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio)*adv
        loss = -(torch.min(ratio*adv, clipped)).mean()
        kl_approx = (old_logp - logp).mean().item()
        return loss, kl_approx

    def update(self):
        obs, act, rew, ret, adv = self.envs.get_data()

        print('UPDATE')

        obs = obs.reshape((-1,) + self.obs_dim).to(self.net.device)
        act = act.reshape(-1).to(self.net.device)
        ret = ret.reshape(-1).to(self.net.device)
        adv = adv.reshape(-1).to(self.net.device)

        with torch.no_grad():
            old_logp = self.net.get_dist(
                obs).log_prob(act)

        for i in range(80):
            self.net.optimizer.zero_grad()

            dist, val = self.net(obs)

            loss_policy, kl = self.compute_policy_gradient(
                dist, act, adv, old_logp)
            loss_value = self.net.criterion(val, ret)

            if kl > 0.05:
                return

            loss = loss_policy + loss_value
            loss.backward()

            self.net.optimizer.step()

    def __enter__(self):
        freeze_support()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.envs.close()

if __name__ == "__main__":
    config = DEFAULT_CONFIG

    with Trainer(config) as trainer:
        trainer.train()
