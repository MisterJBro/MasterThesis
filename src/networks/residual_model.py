import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
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
        self.num_filters = config["num_filters"]
        self.kernel_size = 3
        self.use_se = config["use_se"]
        self.num_res_blocks = config["num_res_blocks"]
        self.size = config["obs_dim"][-1]
        self.MASK_VALUE = -1e4 if config["use_amp"] else -10e8
        self.num_acts = config["num_acts"]

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

        self.opt = optim.Adam(
            list(self.parameters()),
            lr=config["model_lr"],
            weight_decay=config['model_weight_decay']
        )
        #self.scheduler = lr_scheduler.StepLR(
        #    self.opt,
        #    step_size=10,
        #    gamma=0.5,
        #)

        self.device = config["device"]
        self.to(self.device)

    def representation(self, obs):
        return self.repr(obs)

    def dynamics(self, s, a):
        a = F.one_hot(a, num_classes=self.config["num_acts"]).float().reshape(-1, 1, self.size, self.size)
        s = torch.concat((s, a), 1)
        s = self.dyna(s)
        return s

    def prediction_hidden(self, s):
        return self.pred(s)

    def prediction(self, s):
        s = self.pred(s)
        pi = Categorical(logits=self.pred_pi(s).reshape(-1, self.num_acts))
        val = self.pred_val(s).reshape(-1)
        return pi, val

    def prepare_episodes(self, data):
        # Prepare data for training (unrolling each episode)
        act = data["act"]
        done = data["done"]
        ret = data["ret"]
        dist = data["dist"]
        sections = data["sections"]

        episodes = []
        for (start, end) in sections:
            act_ep = []
            val_targets = []
            dist_targets = []
            completed = done[end-1]

            if not completed:
                continue

            for i in range(self.config["model_unroll_len"]):
                act_ep.append(act[start+i:end])
                if i == 0:
                    val_targets.append(ret[start+i:end])
                    dist_targets.append(Categorical(logits=dist[start+i:end]))
                else:
                    val_targets.append(torch.concat((ret[start+i:end], torch.zeros(1, dtype=torch.float32, device=self.device)), 0))
                    dist_targets.append(Categorical(logits=torch.concat((dist[start+i:end], dist[end-1].unsqueeze(0)), 0)))

            episodes.append({
                'start': start,
                'end': end,
                'act_ep': act_ep,
                'val_targets': val_targets,
                'dist_targets': dist_targets,
            })

        return episodes

    def loss(self, data):
        obs = data["obs"]
        scalar_loss = nn.HuberLoss()

        # Get data
        episodes = self.prepare_episodes(data)
        batch_size = int(self.config["num_samples"]/self.config["model_minibatches"])

        # Train model
        for _ in range(self.config["model_iters"]):
            losses = []
            steps = 0
            np.random.shuffle(episodes)
            for ep in episodes:
                start = ep["start"]
                end = ep["end"]
                act_ep = ep["act_ep"]
                val_targets = ep["val_targets"]
                dist_targets = ep["dist_targets"]

                # Predictions
                for i in range(self.config["model_unroll_len"]):
                    if i == 0:
                        state = self.representation(obs[start:end])
                    else:
                        if i != 1:
                            state = state[:-1]
                        state = self.dynamics(state, act_ep[i-1])
                    dist, val = self.prediction(state)

                    loss_val = scalar_loss(val, val_targets[i])
                    loss_dist = kl_divergence(dist_targets[i], dist).mean()
                    loss = loss_val + loss_dist
                    losses.append(loss)

                # Minibatch update
                steps += end - start
                if steps >= batch_size:
                    self.opt.zero_grad()
                    loss = torch.mean(torch.stack(losses))
                    print(loss)
                    loss.backward()
                    self.opt.step()
                    steps = 0
                    losses = []

            if len(losses) != 0:
                self.opt.zero_grad()
                loss = torch.mean(torch.stack(losses))
                loss.backward()
                self.opt.step()
        #self.scheduler.step()

    def save(self, path=f'{PROJECT_PATH}/checkpoints/ve_model.pt'):
        torch.save({
            'parameters': self.state_dict(),
            'optimizer': self.opt.state_dict(),
        }, path)

    def load(self, path=f'{PROJECT_PATH}/checkpoints/ve_model.pt'):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['parameters'])
        self.opt.load_state_dict(checkpoint['optimizer'])
