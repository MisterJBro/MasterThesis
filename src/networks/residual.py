import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import TensorDataset, DataLoader
import pathlib

PROJECT_PATH = pathlib.Path(__file__).parent.parent.parent.absolute().as_posix()


class ResBlock(nn.Module):
    """ Residual Block with Skip Connection, just like ResNet. """
    def __init__(self, config):
        super(ResBlock, self).__init__()
        self.config = config
        self.num_filters = 128
        self.kernel_size = 3

        self.layers = nn.Sequential(
            nn.Conv2d(self.num_filters, self.num_filters, self.kernel_size, padding=1),
            nn.BatchNorm2d(self.num_filters),
            nn.ReLU(),
            nn.Conv2d(self.num_filters, self.num_filters, self.kernel_size, padding=1),
            nn.BatchNorm2d(self.num_filters),
        )

    def forward(self, x):
        return F.relu(x + self.layers(x))


class HexPolicy(nn.Module):
    """Policy network for Hex Env. Deep Convocational Residual Network"""

    def __init__(self, config):
        super(HexPolicy, self).__init__()
        self.config = config
        self.num_filters = 128
        self.kernel_size = 3
        self.num_res_blocks = 19
        self.size = config["obs_dim"][-1]
        self.scalar_loss = nn.HuberLoss()

        # Layers
        self.input_layer = nn.Sequential(
            nn.Conv2d(2, self.num_filters, self.kernel_size, padding=1),
            nn.BatchNorm2d(self.num_filters),
        )
        self.res_blocks = nn.Sequential(*[ResBlock(config) for _ in range(10)])

        # Heads
        self.policy = nn.Sequential(
            nn.Conv2d(self.num_filters, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Flatten(1, -1),
        )
        self.policy_head = nn.Linear(self.size*self.size*16, config["num_acts"])

        self.value = nn.Sequential(
            nn.Conv2d(self.num_filters, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Flatten(1, -1),
            nn.Linear(self.size*self.size*16, 256),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(256, 1)

        self.opt_hidden = optim.Adam(list(self.input_layer.parameters()) + list(self.res_blocks.parameters()), lr=config["pi_lr"])
        self.opt_policy = optim.Adam(list(self.policy.parameters()) + list(self.policy_head.parameters()), lr=config["pi_lr"])
        self.opt_value = optim.Adam(list(self.value.parameters()) + list(self.value_head.parameters()), lr=config["vf_lr"])
        self.device = config["device"]
        self.to(self.device)

    def forward(self, x, legal_actions=None):
        x = self.input_layer(x)
        x = self.res_blocks(x)

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
        x = self.input_layer(x)
        x = self.res_blocks(x)

        logits = self.policy_head(self.policy(x))
        logits = self.filter_actions(logits, legal_actions)
        dist = Categorical(logits=logits)
        return dist

    def get_value(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        val = self.value_head(self.value(x)).reshape(-1)
        return val

    def get_hidden(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return self.policy(x), self.value(x)

    def filter_actions(self, logits, legal_actions=None):
        if legal_actions is None:
            return logits

        # Mask out invalid actions
        new_logits = torch.full(logits.shape, -10e8, dtype=torch.float32).to(self.device)
        for i, row in enumerate(legal_actions):
            new_logits[i, row] = logits[i, row]
        return new_logits

    def loss(self, data):
        obs = data["obs"]
        act = data["act"]
        adv = data["adv"]
        ret = data["ret"]

        # Policy loss
        trainset = TensorDataset(obs, act, adv, ret)
        trainloader = DataLoader(trainset, batch_size=int(self.config["num_samples"]/10), shuffle=True)

        # Minibatch training to fit on GPU memory
        for _ in range(1):
            for obs_batch, act_batch, adv_batch, ret_batch in trainloader:
                self.opt_hidden.zero_grad()
                self.opt_policy.zero_grad()
                self.opt_value.zero_grad()

                dist, val_batch = self(obs_batch)
                loss_value = self.scalar_loss(val_batch, ret_batch)
                logp = dist.log_prob(act_batch)
                loss_policy = -(logp * adv_batch).mean()
                loss_entropy = - dist.entropy().mean()
                loss = loss_policy + self.config["pi_entropy"] * loss_entropy + loss_value
                loss.backward()

                nn.utils.clip_grad_norm_(self.parameters(),  self.config["grad_clip"])
                self.opt_hidden.step()
                self.opt_policy.step()
                self.opt_value.step()

    def loss_gradient(self, data):
        obs = data["obs"]
        act = data["act"]
        adv = data["adv"]

        # Policy loss
        self.opt_policy.zero_grad()
        trainset = TensorDataset(obs, act, adv)
        trainloader = DataLoader(trainset, batch_size=int(self.config["num_samples"]/10))

        # Splitting obs, to fit on GPU memory
        for obs_batch, act_batch, adv_batch in trainloader:
            dist = self.get_dist(obs_batch)
            logp = dist.log_prob(act_batch)
            loss_policy = -(logp * adv_batch).mean()
            loss_entropy = - dist.entropy().mean()
            loss = loss_policy + self.config["pi_entropy"] * loss_entropy
            loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(),  self.config["grad_clip"])
        self.opt_policy.step()

    def loss_value(self, data):
        obs = data["obs"]
        ret = data["ret"]
        scalar_loss = nn.HuberLoss()

        # Value loss
        trainset = TensorDataset(obs, ret)
        trainloader = DataLoader(trainset, batch_size=int(self.config["num_samples"]/self.config["vf_minibatches"]), shuffle=True)

        for _ in range(self.config["vf_iters"]):
            for obs_batch, ret_batch in trainloader:
                self.opt_value.zero_grad()
                val_batch = self.get_value(obs_batch)
                loss_value = scalar_loss(val_batch, ret_batch)
                loss_value.backward()
                nn.utils.clip_grad_norm_(self.value.parameters(),  self.config["grad_clip"])
                self.opt_value.step()

    def save(self, path=f'{PROJECT_PATH}/checkpoints/policy_pdlm.pt'):
        torch.save({
            'parameters': self.state_dict(),
            'optimizer_policy': self.opt_policy.state_dict(),
            'optimizer_value': self.opt_value.state_dict(),
        }, path)

    def load(self, path=f'{PROJECT_PATH}/checkpoints/policy_pdlm.pt'):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['parameters'])
        self.opt_policy.load_state_dict(checkpoint['optimizer_policy'])
        self.opt_value.load_state_dict(checkpoint['optimizer_value'])