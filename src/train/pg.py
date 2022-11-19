import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, ChainDataset, ConcatDataset, DataLoader
import humanize
import numpy as np
from src.train.trainer import Trainer
import os
import time
import pathlib

PROJECT_PATH = pathlib.Path(__file__).parent.parent.parent.absolute().as_posix()


class PGTrainer(Trainer):
    """ Train a policy using Policy Gradient with baseline."""

    def update(self, sample_batch):
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

    def update(self, eps):
        # Config
        batch_size = 2048
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
        num_batch_splits = np.ceil(obs.shape[0] / batch_size)

        # Policy loss
        trainset = TensorDataset(obs, act, legal_act)
        trainloader = DataLoader(trainset, batch_size=batch_size, pin_memory=True)

        # Get value and old log_p
        with torch.no_grad():
            logp = []
            val = []
            for obs_bt, act_bt, legal_act_bt in trainloader:
                obs_bt = obs_bt.to(device)
                act_bt = act_bt.to(device)
                legal_act_bt = legal_act_bt.to(device)

                dist_bt, val_bt = self.policy(obs_bt, legal_actions=legal_act_bt)
                logp_bt = dist_bt.log_prob(act_bt)
                logp.append(logp_bt)
                val.append(val_bt)
            old_logp = torch.cat(logp, 0).cpu()
            val = torch.cat(val, 0).cpu()
        #start = time.time()
        #torch.cuda.synchronize()

        # Advantage estimation
        adv = ret - val

        trainset = TensorDataset(obs, act, legal_act, ret, old_logp, adv)
        trainloader = DataLoader(trainset, batch_size=batch_size, pin_memory=True)

        # Minibatch training to fit on GPU memory
        for _ in range(self.config["ppo_iters"]):
            self.policy.optim.zero_grad(set_to_none=True)
            for obs_bt, act_bt, legal_act_bt, ret_bt, old_logp_bt, adv_bt in trainloader:
                obs_bt = obs_bt.to(device)
                act_bt = act_bt.to(device)
                legal_act_bt = legal_act_bt.to(device)
                ret_bt = ret_bt.to(device)
                old_logp_bt = old_logp_bt.to(device)
                adv_bt = adv_bt.to(device)

                with torch.autocast(device_type=self.config["amp_device"], enabled=self.config["use_amp"]):
                    dist_bt, val_bt = self.policy(obs_bt, legal_actions=legal_act_bt)

                    # PPO loss
                    logp_bt = dist_bt.log_prob(act_bt)
                    ratio = torch.exp(logp_bt - old_logp_bt)
                    clipped = torch.clamp(ratio, 1-self.config["clip_ratio"], 1+self.config["clip_ratio"])*adv_bt
                    loss_policy = -(torch.min(ratio*adv_bt, clipped)).mean()

                    loss_entropy = - dist_bt.entropy().mean()
                    loss_value = self.scalar_loss(val_bt, ret_bt)
                    loss = loss_policy + self.config["pi_entropy"] * loss_entropy + self.config["vf_scale"] * loss_value
                    loss /= num_batch_splits

                    self.log("loss_pi", loss_policy.item(), show=False)
                    self.log("loss_entr", loss_entropy.item(), show=True)
                    self.log("loss_v", loss_value.item(), show=False)

                # AMP loss backward
                self.scaler.scale(loss).backward()

            # AMP Update
            self.scaler.unscale_(self.policy.optim)
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.config["grad_clip"])
            self.scaler.step(self.policy.optim)
            self.scaler.update()

            #mem = torch.cuda.mem_get_info()
            #print(f"GPU Memory: {humanize.naturalsize(mem[0])} / {humanize.naturalsize(mem[1])}")


class PPOTrainerModel(Trainer):
    """ Train a policy using Proximal Policy Gradient."""

    def __init__(self, config, policy, model):
        super().__init__(config)
        self.policy = policy
        self.model = model

    def update(self, sample_batch):
        data = sample_batch.to_tensor_dict()
        self.model.loss(data)

    def checkpoint(self, iter):
        last_path = f'{PROJECT_PATH}/checkpoints/model_{str(self.config["env"]).lower()}_{self.__class__.__name__.lower().replace("trainer","")}_iter={iter}_metric={self.log[self.config["log_main_metric"]]:.0f}.pt'
        self.save_paths.append(last_path)
        if len(self.save_paths) > self.config["num_checkpoints"]:
            path = self.save_paths.pop(0)
            if os.path.isfile(path):
                os.remove(path)
            else:
                print("Warning: %s checkpoint not found" % path)
        self.save(path=last_path)

    def save(self, path=None):
        if path is not None:
            self.model.save(path=path)
        else:
            self.model.save()
