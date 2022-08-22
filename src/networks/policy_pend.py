import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import TensorDataset, DataLoader
import pathlib

PROJECT_PATH = pathlib.Path(__file__).parent.parent.parent.absolute().as_posix()


class PendulumPolicy(nn.Module):
    """Policy network for Pendulum Env."""

    def __init__(self, config):
        super(PendulumPolicy, self).__init__()
        self.config = config
        hidden_size = 1024

        #self.hidden = nn.Sequential(
        #    nn.Linear(config["flat_obs_dim"], hidden_size),
        #    nn.ReLU(),
        #    nn.Linear(hidden_size, hidden_size),
        #    nn.ReLU(),
        #)
        self.hidden = nn.Identity()

        # Heads
        self.policy = nn.Sequential(
            nn.Linear(config["flat_obs_dim"], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, config["num_acts"]),
        )
        self.value = nn.Sequential(
            nn.Linear(config["flat_obs_dim"], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        #self.opt = optim.Adam(self.parameters(), lr=config["pi_lr"])
        self.opt_policy = optim.Adam(list(self.hidden.parameters())+ list(self.policy.parameters()), lr=config["pi_lr"])
        self.opt_value = optim.Adam(list(self.value.parameters()), lr=config["vf_lr"])
        self.device = config["device"]
        self.to(self.device)

    def forward(self, x):
        x = self.hidden(x)
        dist = Categorical(logits=self.policy(x))
        val = self.value(x).reshape(-1)
        return dist, val

    def get_action(self, x_numpy):
        with torch.no_grad():
            x = torch.as_tensor(x_numpy, dtype=torch.float32).to(self.device)
            dist = self.get_dist(x)
            act = dist.sample()

        return act.cpu().numpy()

    def get_dist(self, x):
        x = self.hidden(x)
        dist = Categorical(logits=self.policy(x))
        return dist

    def get_value(self, x):
        x = self.hidden(x)
        val = self.value(x).reshape(-1)
        return val

    def loss_gradient(self, data):
        obs = data["obs"]
        act = data["act"]
        adv = data["adv"]

        # Policy loss
        self.opt_policy.zero_grad()

        dist = self.get_dist(obs)
        logp = dist.log_prob(act)
        loss_policy = -(logp * adv).mean()
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

    def save(self, path=f'{PROJECT_PATH}/checkpoints/pendulum_policy.pt'):
        torch.save({
            'parameters': self.state_dict(),
            'optimizer_policy': self.opt_policy.state_dict(),
            'optimizer_value': self.opt_value.state_dict(),
        }, path)

    def load(self, path=f'{PROJECT_PATH}/checkpoints/pendulum_policy.pt'):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['parameters'])
        self.opt_policy.load_state_dict(checkpoint['optimizer_policy'])
        self.opt_value.load_state_dict(checkpoint['optimizer_value'])