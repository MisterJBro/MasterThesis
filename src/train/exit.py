import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.networks.policy_pend import PendulumPolicy
from src.search.alpha_zero import AlphaZero
from src.search.state import State

from torch.distributions import Categorical
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions.kl import kl_divergence
from src.train.trainer import Trainer


class AZExitTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

        self.num_acts = config["num_acts"]
        self.policy = PendulumPolicy(config)
        self.az = AlphaZero(self.policy, config)

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

    def get_action(self, obs, envs, use_best=False):
        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        if envs is None:
            envs = self.envs.get_all_env()
        states = [State(env, obs=obs[i]) for i, env in enumerate(envs)]

        with torch.no_grad():
            dist = self.policy.get_dist(obs)
        logits = dist.logits.cpu()

        q = self.az.distributed_search(states)
        dist = F.softmax(logits + 10.0*q, dim=-1)
        if use_best:
            act = torch.max(dist).numpy()
        else:
            act = Categorical(probs=dist).sample().numpy()

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

        for i in range(1):
            for obs_batch, target_batch, ret_batch in trainloader:
                self.policy.opt_policy.zero_grad()
                self.policy.opt_value.zero_grad()

                dist_batch, val_batch = self.policy(obs_batch)
                loss_dist = kl_divergence(Categorical(probs=target_batch), dist_batch).mean()
                loss_value = scalar_loss(val_batch, ret_batch)
                loss = loss_dist + loss_value
                loss.backward()

                #nn.utils.clip_grad_norm_(self.policy.parameters(),  self.config["grad_clip"])
                self.policy.opt_policy.step()
                self.policy.opt_value.step()
