import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


    def get_action(self, obs, env_list=None, use_best=False, legal_actions=None):
        if env_list is None:
            env_list = self.envs.get_all_env()
        states = [State(env, obs=obs[i]) for i, env in enumerate(env_list)]
        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)

        # Policy inference
        with torch.no_grad():
            dist = self.policy.get_dist(obs, legal_actions=legal_actions)
        logits = dist.logits.cpu()

        # Search best action distribution
        result = self.search_algo.search(states)
        if self.config["search_return_adv"]:
            dist = F.softmax(logits + result, dim=-1)
        else:
            dist = result

        # Sample or argmax action
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
                self.policy.opt_hidden.zero_grad(set_to_none=True)
                self.policy.opt_policy.zero_grad(set_to_none=True)
                self.policy.opt_value.zero_grad(set_to_none=True)

                dist_batch, val_batch = self.policy(obs_batch)
                loss_dist = kl_divergence(Categorical(probs=target_batch), dist_batch).mean()
                loss_value = self.scalar_loss(val_batch, ret_batch)
                loss = loss_dist + loss_value
                loss.backward()

                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config["grad_clip"])
                self.policy.opt_hidden.step()
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
