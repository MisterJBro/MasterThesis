import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.networks.policy_pend import PendulumPolicy
from src.search.alpha_zero.alpha_zero import AlphaZero
from src.search.state import State

from torch.distributions import Categorical
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions.kl import kl_divergence
from src.train.trainer import Trainer


class ExitTrainer(Trainer):
    """ Wrapper for Expert Iteration. Using a search algorithm as expert e.g.
    AlphaZero, then do the following iterative process:
    - search the currently best action
    - distill the action into the policy
    """

    def __init__(self, config, search_algo, policy, model=None):
        super().__init__(config)

        self.num_acts = config["num_acts"]
        self.scalar_loss = nn.HuberLoss()
        self.search_algo = search_algo
        self.policy = policy
        self.model = model

    def get_action(self, obs, envs=None, use_best=False):
        if envs is None:
            envs = self.envs.get_all_env()
        states = [State(env, obs=obs[i]) for i, env in enumerate(envs)]
        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            dist = self.policy.get_dist(obs)
        logits = dist.logits.cpu()

        scaled_adv = self.search_algo.search(states)
        dist = F.softmax(logits + scaled_adv, dim=-1)
        if use_best:
            act = torch.argmax(dist).numpy()
        else:
            act = Categorical(probs=dist).sample().numpy()

        return act, dist

    def update(self, sample_batch):
        data = sample_batch.to_tensor_dict()
        obs = data["obs"]
        dist = data["dist"]
        ret = data["ret"]

        # Distill planning targets into policy
        trainset = TensorDataset(obs, dist, ret)
        trainloader = DataLoader(trainset, batch_size=int(self.config["num_samples"]/10), shuffle=True)

        for _ in range(3):
            for obs_batch, target_batch, ret_batch in trainloader:
                self.policy.opt_policy.zero_grad()
                self.policy.opt_value.zero_grad()

                dist_batch, val_batch = self.policy(obs_batch)
                loss_dist = kl_divergence(Categorical(probs=target_batch), dist_batch).mean()
                loss_value = self.scalar_loss(val_batch, ret_batch)
                loss = loss_dist + loss_value
                loss.backward()

                nn.utils.clip_grad_norm_(self.policy.parameters(),  self.config["grad_clip"])
                self.policy.opt_policy.step()
                self.policy.opt_value.step()

        # Model training
        if self.model is not None:
            with torch.no_grad():
                data["logits"] = self.policy.get_dist(obs).logits

            self.model.loss(data)

            # Update policy
            self.search_algo.update(self.policy.state_dict(), self.model.state_dict())

        else:
            self.search_algo.update(self.policy.state_dict())

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self.search_algo.close()


class AZExitTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

        self.num_acts = config["num_acts"]
        self.dist_q_scale = config["az_dist_q_scale"]
        self.policy = PendulumPolicy(config)
        self.az = AlphaZero(self.policy, config)

    def get_action(self, obs, envs=None, use_best=False):
        if envs is None:
            envs = self.envs.get_all_env()
        states = [State(env, obs=obs[i]) for i, env in enumerate(envs)]
        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            dist = self.policy.get_dist(obs)
        logits = dist.logits.cpu()

        q = self.az.distributed_search(states)
        dist = F.softmax(logits + self.dist_q_scale*q, dim=-1)
        if use_best:
            act = torch.argmax(dist).numpy()
        else:
            act = Categorical(probs=dist).sample().numpy()

        return act, dist

    def update(self, sample_batch):
        # Check model performance
        #if self.config["az_force_improvement"] and self.log["avg ret"] < self.log.best_metric:
        #   self.policy.load(self.log.best_model_path)

        data = sample_batch.to_tensor_dict()
        obs = data["obs"]
        dist = data["dist"]
        ret = data["ret"]
        scalar_loss = nn.HuberLoss()

        # Distill planning targets into policy
        trainset = TensorDataset(obs, dist, ret)
        trainloader = DataLoader(trainset, batch_size=int(self.config["num_samples"]/self.config["az_dist_minibatches"]), shuffle=True)

        for _ in range(self.config["az_dist_iters"]):
            for obs_batch, target_batch, ret_batch in trainloader:
                self.policy.opt_policy.zero_grad()
                self.policy.opt_value.zero_grad()

                dist_batch, val_batch = self.policy(obs_batch)
                loss_dist = kl_divergence(Categorical(probs=target_batch), dist_batch).mean()
                loss_value = scalar_loss(val_batch, ret_batch)
                loss = loss_dist + loss_value
                loss.backward()

                nn.utils.clip_grad_norm_(self.policy.parameters(),  self.config["grad_clip"])
                self.policy.opt_policy.step()
                self.policy.opt_value.step()

        # Update AlphaZero policy
        self.az.update(self.policy.state_dict())

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self.az.close()