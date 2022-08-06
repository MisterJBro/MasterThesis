import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import pathlib

PROJECT_PATH = pathlib.Path(__file__).parent.absolute().as_posix()


class ActorCriticPolicy(nn.Module):
    """Actor-Critic Policy network"""

    def __init__(self, config):
        super(ActorCriticPolicy, self).__init__()
        hidden_size = 512

        self.hidden = nn.Sequential(
            nn.Linear(config["flat_obs_dim"],hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

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
        self.opt_policy = optim.Adam(list(self.policy.parameters()), lr=config["pi_lr"])
        self.opt_value = optim.Adam(list(self.value.parameters()), lr=config["vf_lr"])
        self.device = config["device"]
        self.to(self.device)

    def forward(self, x):
        #x = self.hidden(x)
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
        #x = self.hidden(x)
        dist = Categorical(logits=self.policy(x))
        return dist

    def get_value(self, x):
        #x = self.hidden(x)
        val = self.value(x).reshape(-1)
        return val

    def save(self, path=f'{PROJECT_PATH}/checkpoints/policy.pt'):
        torch.save({
            'parameters': self.state_dict(),
            'optimizer_policy': self.opt_policy.state_dict(),
            'optimizer_value': self.opt_value.state_dict(),
        }, path)

    def load(self, path=f'{PROJECT_PATH}/checkpoints/policy.pt'):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['parameters'])
        self.opt_policy.load_state_dict(checkpoint['optimizer_policy'])
        self.opt_value.load_state_dict(checkpoint['optimizer_value'])