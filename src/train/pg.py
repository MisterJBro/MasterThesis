import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import humanize
import numpy as np
from src.train.trainer import Trainer

class PGTrainer(Trainer):
    """ Train a policy using Policy Gradient with baseline."""

    def __init__(self, config, policy):
        super().__init__(config)
        self.policy = policy

    def update(self, sample_batch):
        self.policy.train()
        data = sample_batch.to_tensor_dict()
        obs = data["obs"]
        act = data["act"]
        ret = data["ret"]
        val = data["val"]
        adv = ret
        data["adv"] = adv

        trainset = TensorDataset(obs, act, adv, ret)
        trainloader = DataLoader(trainset, batch_size=int(self.config["num_samples"]/self.config["num_batch_split"]))

        # Minibatch training to fit on GPU memory
        for _ in range(self.config["pg_iters"]):
            self.policy.optim.zero_grad(set_to_none=True)
            for obs_batch, act_batch, adv_batch, ret_batch in trainloader:
                with torch.autocast(device_type=self.config["amp_device"], enabled=self.config["use_amp"]):
                    dist, val_batch = self.policy(obs_batch)

                    # PG loss
                    logp = dist.log_prob(act_batch)
                    loss_policy = -(logp * adv_batch).mean()
                    loss_entropy = - dist.entropy().mean()
                    loss_value = self.scalar_loss(val_batch, ret_batch)
                    loss = loss_policy + self.config["pi_entropy"] * loss_entropy + self.config["vf_scale"] * loss_value
                    loss /= self.config["num_batch_split"]

                # AMP loss backward
                self.scaler.scale(loss).backward()

            # AMP Update
            self.scaler.unscale_(self.policy.optim)
            nn.utils.clip_grad_norm_(self.policy.parameters(),  self.config["grad_clip"])
            self.scaler.step(self.policy.optim)
            self.scaler.update()

            #mem = torch.cuda.mem_get_info()
            #print(f"GPU Memory: {humanize.naturalsize(mem[0])} / {humanize.naturalsize(mem[1])}")


class PPOTrainer(PGTrainer):
    """ Train a policy using Proximal Policy Gradient."""

    def update(self, sample_batch):
        self.policy.train()
        #self.policy.set_requires_grad(True)
        data = sample_batch.to_tensor_dict()
        obs = data["obs"]
        act = data["act"]
        ret = data["ret"]
        val = data["val"]
        legal_act = data["legal_act"]
        adv = ret - val
        data["adv"] = adv

        # Policy loss
        trainset = TensorDataset(obs, act)
        trainloader = DataLoader(trainset, batch_size=int(self.config["num_samples"]/self.config["num_batch_split"]))

        # Get old log_p
        with torch.no_grad():
            old_logp = []
            for obs_batch, act_batch in trainloader:
                old_logp_batch = self.policy.get_dist(obs_batch,  legal_actions=legal_act).log_prob(act_batch)
                old_logp.append(old_logp_batch)
            old_logp = torch.cat(old_logp, 0)

        trainset = TensorDataset(obs, act, adv, ret, old_logp)
        trainloader = DataLoader(trainset, batch_size=int(self.config["num_samples"]/self.config["num_batch_split"]))

        # Minibatch training to fit on GPU memory
        for _ in range(self.config["ppo_iters"]):
            self.policy.optim.zero_grad(set_to_none=True)
            for obs_batch, act_batch, adv_batch, ret_batch, old_logp_batch in trainloader:
                with torch.autocast(device_type=self.config["amp_device"], enabled=self.config["use_amp"]):
                    dist, val_batch = self.policy(obs_batch,  legal_actions=legal_act)

                    # PPO loss
                    logp = dist.log_prob(act_batch)
                    ratio = torch.exp(logp - old_logp_batch)
                    clipped = torch.clamp(ratio, 1-self.config["clip_ratio"], 1+self.config["clip_ratio"])*adv_batch
                    loss_policy = -(torch.min(ratio*adv_batch, clipped)).mean()
                    #kl_approx = (old_logp_batch - logp).mean().item()
                    #if kl_approx > self.config["kl_approx_max"]:
                    #    self.policy.set_requires_grad(False)
                    loss_entropy = - dist.entropy().mean()
                    loss_value = self.scalar_loss(val_batch, ret_batch)
                    loss = loss_policy + self.config["pi_entropy"] * loss_entropy + self.config["vf_scale"] * loss_value
                    loss /= self.config["num_batch_split"]
                    #self.log("kl_approx", kl_approx)
                    self.log("loss_entropy", loss_entropy.item())
                    self.log("loss_policy", loss_policy.item())
                    self.log("loss_value", loss_value.item())
                    #print("\t", np.round(loss_value.item(), 3))

                # AMP loss backward
                self.scaler.scale(loss).backward()

            # AMP Update
            self.scaler.unscale_(self.policy.optim)
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.config["grad_clip"])
            self.scaler.step(self.policy.optim)
            self.scaler.update()

            #mem = torch.cuda.mem_get_info()
            #print(f"GPU Memory: {humanize.naturalsize(mem[0])} / {humanize.naturalsize(mem[1])}")
