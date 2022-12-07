import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pathlib

PROJECT_PATH = pathlib.Path(__file__).parent.parent.parent.absolute().as_posix()

class SEBlock(nn.Module):
    """Squeeze and Excitation Block from https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResBlock(nn.Module):
    """ Residual Block with Skip Connection, just like ResNet. """
    def __init__(self, num_filters, kernel_size, use_se):
        super(ResBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, kernel_size, padding=1),
            nn.BatchNorm2d(num_filters),
        )
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(num_filters)

    def forward(self, x):
        out = self.layers(x)
        if self.use_se:
            out = self.se(out)
        return F.relu(x + out)


class ValueEquivalenceModel(nn.Module):
    """Value Equivalence Model."""

    def __init__(self, config):
        super(ValueEquivalenceModel, self).__init__()
        self.config = config
        self.num_filters = config["model_num_filters"]
        self.kernel_size = 3
        self.use_se = config["use_se"]
        self.num_res_blocks = config["model_num_res_blocks"]
        self.size = config["obs_dim"][-1]
        self.MASK_VALUE = -1e4 if config["use_amp"] else -10e8
        self.num_acts = config["num_acts"]
        self.batch_size = config["batch_size"]
        self.zero_rew = True

        # Representation function h
        self.repr = nn.Sequential(
            nn.Conv2d(2, self.num_filters, kernel_size=self.kernel_size, padding=1),
            nn.BatchNorm2d(self.num_filters),
            nn.ReLU(inplace=True),
            *[ResBlock(self.num_filters, self.kernel_size, self.use_se) for _ in range(self.num_res_blocks)],
        )

        # Dynamic function g
        self.dyna = nn.Sequential(
            nn.Conv2d(self.num_filters+1, self.num_filters, 1),
            *[ResBlock(self.num_filters, self.kernel_size, self.use_se) for _ in range(self.num_res_blocks)],
        )

        # Prediction functions f, for policy, value and reward functions
        self.pred = nn.Sequential(
            *[ResBlock(self.num_filters, self.kernel_size, self.use_se) for _ in range(self.num_res_blocks)],
        )

        self.pred_val = nn.Sequential(
            nn.Conv2d(self.num_filters, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(1, -1),
            nn.Linear(self.size*self.size*32, 1),
            nn.Tanh(),
        )
        self.pred_pi = nn.Sequential(
            nn.Conv2d(self.num_filters, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(1, -1),
            nn.Linear(self.size*self.size*32, config["num_acts"])
        )
        self.pred_rew = nn.Sequential(
            nn.Conv2d(self.num_filters, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(1, -1),
            nn.Linear(self.size*self.size*32, 1),
            nn.Tanh(),
        )

        self.device = config["device"]
        self.to(self.device)

    def representation(self, obs):
        return self.repr(obs)

    def dynamics(self, s, a):
        a = F.one_hot(a, num_classes=self.config["num_acts"]).float().reshape(-1, 1, self.size, self.size)
        s = torch.concat((s, a), 1)
        s = self.dyna(s)
        if self.zero_rew:
            rew = torch.zeros(s.shape[0], dtype=torch.float32, device=self.device)
        else:
            rew = self.pred_rew(s).reshape(-1)
        return s, rew

    def prediction_hidden(self, s):
        return self.pred(s)

    def prediction(self, s):
        s = self.pred(s)
        pi = Categorical(logits=self.pred_pi(s).reshape(-1, self.num_acts))
        val = self.pred_val(s).reshape(-1)
        return pi, val

    def prepare_eps(self, eps, vals):
        obs = torch.as_tensor(np.concatenate([e.obs for e in eps], 0))
        val_target = []
        dist_target = []
        act = []
        rew_target = []

        for e, v_t in zip(eps, vals):
            # Get data
            ret_ep = []
            dist_ep = []
            act_ep = []
            rew_ep = []

            for i in range(self.config["model_unroll_len"]):
                v = np.concatenate([v_t[i:], np.full((i), v_t[-1], dtype=np.float32)], 0)
                ret_ep.append(v)
                d = np.concatenate([e.dist[i:], np.zeros((i, self.num_acts), dtype=np.float32)], 0)
                dist_ep.append(d)

                if i < self.config["model_unroll_len"] - 1:
                    a = np.concatenate([e.act[i:], np.random.randint(self.num_acts, size=i)], 0, dtype=np.int64)
                    act_ep.append(a)
                    r = np.concatenate([e.rew[i:], np.zeros((i), dtype=np.float32)], 0)
                    rew_ep.append(r)

            # Add
            val_target.append(np.stack(ret_ep, 1))
            dist_target.append(np.stack(dist_ep, 1))
            act.append(np.stack(act_ep, 1))
            rew_target.append(np.stack(rew_ep, 1))

        # To tensors
        val_target = torch.as_tensor(np.concatenate(val_target, 0))
        dist_target = torch.as_tensor(np.concatenate(dist_target, 0))
        act = torch.as_tensor(np.concatenate(act, 0))
        rew_target = torch.as_tensor(np.concatenate(rew_target, 0))

        # Obs: (N, 2, size, size)  Val: (N, model_len)  Dist: (N, model_len, num_acts), Act: (N, model_len-1), Rew: (N, model_len-1)
        return {
            'obs': obs,
            'val_target': val_target,
            'dist_target': dist_target,
            'act': act,
            'rew_target': rew_target,
        }

    def loss(self, eps, vals):
        # Init optimizer and scheduler
        self.opt = optim.Adam(
            list(self.parameters()),
            lr=self.config["model_lr"],
            weight_decay=self.config['model_weight_decay']
        )
        self.scheduler = lr_scheduler.StepLR(
            self.opt,
            step_size=300,
            gamma=0.5,
        )

        # Get data
        batch_size = self.config["model_batch_size"]
        data = self.prepare_eps(eps, vals)
        trainset = TensorDataset(data['obs'], data['val_target'], data['dist_target'], data['act'], data['rew_target'])
        trainloader = DataLoader(trainset, batch_size=batch_size, pin_memory=True, shuffle=True)

        # Train model
        iter = 0
        losses = []
        for _ in range(self.config["model_iters"]):
            for obs, val_target, dist_target, act, rew_target in trainloader:
                obs = obs.to(self.device)
                val_target = val_target.to(self.device)
                dist_target = dist_target.to(self.device)
                act = act.to(self.device)
                rew_target = rew_target.to(self.device)

                # Predictions
                for i in range(self.config["model_unroll_len"]):
                    if i == 0:
                        state = self.representation(obs)
                        loss_rew = 0
                    else:
                        state, rew = self.dynamics(state, act[:, i-1])
                        loss_rew = 0#F.mse_loss(rew, rew_target[:, i-1].reshape(-1))
                    dist, val = self.prediction(state)

                    loss_val = F.mse_loss(val, val_target[:, i].reshape(-1))
                    loss_dist = kl_divergence(Categorical(logits=dist_target[:, i]), dist).mean()
                    loss = loss_val + loss_dist + loss_rew
                    losses.append(loss)

                # Update
                if (iter+1) % self.config["acc_grads"] == 0:
                    loss = torch.mean(torch.stack(losses))
                    print(f"Iter {iter}  Loss: {loss.item():.04f}")
                    loss.backward()
                    self.opt.step()
                    self.scheduler.step()
                    self.opt.zero_grad()
                    losses = []
                iter += 1

    def test(self, eps, vals):
        with torch.inference_mode():
            # Get data
            batch_size = self.config["model_batch_size"]
            data = self.prepare_eps(eps, vals)
            trainset = TensorDataset(data['obs'], data['val_target'], data['dist_target'], data['act'], data['rew_target'])
            trainloader = DataLoader(trainset, batch_size=batch_size, pin_memory=True)

            # Train model
            losses = []
            for obs, val_target, dist_target, act, rew_target in trainloader:
                obs = obs.to(self.device)
                val_target = val_target.to(self.device)
                dist_target = dist_target.to(self.device)
                act = act.to(self.device)
                rew_target = rew_target.to(self.device)

                # Predictions
                for i in range(self.config["model_unroll_len"]):
                    if i == 0:
                        state = self.representation(obs)
                        loss_rew = 0
                    else:
                        state, rew = self.dynamics(state, act[:, i-1])
                        loss_rew = 0#F.mse_loss(rew, rew_target[:, i-1].reshape(-1))
                    dist, val = self.prediction(state)

                    loss_val = F.mse_loss(val, val_target[:, i].reshape(-1))
                    loss_dist = kl_divergence(Categorical(logits=dist_target[:, i]), dist).mean()
                    loss = loss_val + loss_dist + loss_rew
                    losses.append(loss)
            loss = torch.mean(torch.stack(losses)).item()
        return loss

    def save(self, path=f'{PROJECT_PATH}/checkpoints/ve_model.pt'):
        torch.save({
            'parameters': self.state_dict(),
            'optimizer': self.opt.state_dict(),
        }, path)

    def load(self, path=f'{PROJECT_PATH}/checkpoints/ve_model.pt'):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['parameters'])
        self.opt = optim.Adam(
            list(self.parameters()),
            lr=self.config["model_lr"],
            weight_decay=self.config['model_weight_decay']
        )
        self.opt.load_state_dict(checkpoint['optimizer'])
