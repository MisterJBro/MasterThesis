import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
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
            nn.LeakyReLU(inplace=True),
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


class HexPolicy(nn.Module):
    """Policy network for Hex Env. Deep Convocational Residual Network"""

    def __init__(self, config):
        super(HexPolicy, self).__init__()
        self.config = config
        self.num_filters = config["num_filters"]
        self.kernel_size = 3
        self.use_se = config["use_se"]
        self.num_res_blocks = config["num_res_blocks"]
        self.size = config["obs_dim"][-1]

        # Layers
        self.body = nn.Conv2d(2, self.num_filters, kernel_size=self.kernel_size, padding=1)
            #nn.Sequential(
            #,
            #nn.BatchNorm2d(self.num_filters),
            #nn.LeakyReLU(inplace=True),
            #*[ResBlock(self.num_filters, self.kernel_size, self.use_se) for _ in range(self.num_res_blocks)],
        #)

        # Heads
        self.policy = nn.Sequential(
            nn.Conv2d(self.num_filters, 16, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(1, -1),
        )
        self.policy_head = nn.Linear(self.size*self.size*16, config["num_acts"])

        self.value = nn.Sequential(
            nn.Conv2d(self.num_filters, 16, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(1, -1),
            nn.Linear(self.size*self.size*16, 256),
            nn.LeakyReLU(inplace=True),
        )
        self.value_head = nn.Sequential(
            nn.Linear(256, 1),
            nn.Tanh(),
        )

        self.optim = optim.Adam(self.parameters(), lr=config["pi_lr"])
        self.device = config["device"]
        self.to(self.device)

    def forward(self, x, legal_actions=None):
        x = torch.randn(1, 2, 9, 9).float().cuda()
        print("Before: ", x.sum().item())
        x = self.body(x)
        print("After: ", x.sum().item())

        logits = self.policy_head(self.policy(x))
        logits = self.filter_actions(logits, legal_actions)
        dist = Categorical(logits=logits)
        val = self.value_head(self.value(x)).reshape(-1)
        return dist, val

    def get_action(self, x_numpy, legal_actions=None):
        with torch.no_grad():
            x = torch.as_tensor(x_numpy, dtype=torch.float32).to(self.device)
            dist = self.get_dist(x, legal_actions=legal_actions)
            act = dist.sample()

        return act.cpu().numpy()

    def get_dist(self, x, legal_actions=None):
        x = self.body(x)

        logits = self.policy_head(self.policy(x))
        logits = self.filter_actions(logits, legal_actions)
        dist = Categorical(logits=logits)
        return dist

    def get_value(self, x):
        x = self.body(x)
        val = self.value_head(self.value(x)).reshape(-1)
        return val

    def get_hidden(self, x):
        x = self.body(x)
        return self.policy(x), self.value(x)

    def filter_actions(self, logits, legal_actions=None):
        if legal_actions is None:
            return logits

        # Mask out invalid actions
        MASK_VALUE = -10e8 if logits.dtype == torch.float32 else -1e4
        new_logits = torch.full(logits.shape, MASK_VALUE, dtype=logits.dtype).to(self.device)
        for i, row in enumerate(legal_actions):
            new_logits[i, row] = logits[i, row]
        return new_logits

    def save(self, path=f'{PROJECT_PATH}/checkpoints/policy.pt'):
        torch.save({
            'parameters': self.state_dict(),
            'optimizer': self.optim.state_dict(),
        }, path)

    def load(self, path=f'{PROJECT_PATH}/checkpoints/policy.pt'):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['parameters'])
        self.optim.load_state_dict(checkpoint['optimizer'])