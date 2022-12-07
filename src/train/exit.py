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

    def __init__(self, config, policy, search_algo, model=None):
        super().__init__(config, policy, search_algo=search_algo, model=model)
        self.num_acts = config["num_acts"]

    def get_self_play_actions(self, policies, mapping, obs, info, use_best=False):
        env_list = self.envs.get_envs(eid)
        states = [State(env, obs=obs[i]) for i, env in enumerate(env_list)]
        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        legal_act = info["legal_act"]
        eid = info["eid"]
        pid = info["pid"]
        pol_id = mapping[eid, pid]

        # CUDA inference (async)
        eids, dists, pol_ids = [], [], []
        for p, policy in enumerate(policies):
            is_p = pol_id == p
            if np.sum(is_p) > 0:
                obs_p = obs[is_p]
                legal_act_p = legal_act[is_p]
                with torch.inference_mode():
                    dist_p = policy.get_dist(obs_p, legal_actions=legal_act_p)
                dists.append(dist_p)
                eids.append(eid[is_p])
                pol_ids.append(np.full(np.sum(is_p), p))

        # Build actions
        act = []
        eid = []
        pol_id = []
        dist = []
        for eid_p, dist_p, pol_id_p in zip(eids, dists, pol_ids):
            if use_best:
                act_p = dist_p.logits.argmax(-1)
            else:
                act_p = dist_p.sample()
            act_p = [a.cpu().numpy().item() for a in act_p]
            act += act_p
            eid.append(eid_p)
            pol_id.append(pol_id_p)
            dist.append(dist_p.logits.cpu().numpy())
        return act, np.concatenate(eid, 0), np.concatenate(dist, 0), np.concatenate(pol_id, 0)

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

    def update(self, eps):
        self.policy.train()
        # Config
        batch_size = self.config["batch_size"]
        device = self.config["device"]

        # Get data
        obs = torch.as_tensor(np.concatenate([e.obs for e in eps], 0))
        act = torch.as_tensor(np.concatenate([e.act for e in eps], 0, dtype=np.int32))
        ret = torch.as_tensor(np.concatenate([e.ret for e in eps], 0))
        legal_act = torch.as_tensor(np.concatenate([e.legal_act for e in eps], 0))

        # Filter by policies
        pol_id = torch.as_tensor(np.concatenate([e.pol_id for e in eps], 0, dtype=np.int32))
        obs = obs[pol_id == 0]
        act = act[pol_id == 0]
        ret = ret[pol_id == 0]
        legal_act = legal_act[pol_id == 0]

        # Policy loss
        trainset = TensorDataset(obs, act, legal_act)
        trainloader = DataLoader(trainset, batch_size=batch_size, pin_memory=True)

        # Distill planning targets into policy
        trainset = TensorDataset(obs, dist, ret)
        trainloader = DataLoader(trainset, batch_size=int(self.config["num_samples"]/10), shuffle=True)

        for _ in range(3):
            for obs_batch, target_batch, ret_batch in trainloader:
                self.policy.optim.zero_grad(set_to_none=True)

                dist_batch, val_batch = self.policy(obs_batch)
                loss_dist = kl_divergence(Categorical(probs=target_batch), dist_batch).mean()
                loss_value = self.scalar_loss(val_batch, ret_batch)
                loss = loss_dist + loss_value
                loss.backward()

                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config["grad_clip"])
                self.policy.optim.step()

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
