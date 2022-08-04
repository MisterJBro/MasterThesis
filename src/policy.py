import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCriticPolicy(nn.Module):
    """Actor-Critic Policy network"""

    def __init__(self, config):
        super(ActorCriticPolicy, self).__init__()

        self.hidden = nn.Sequential(
            nn.Linear(config["obs_dim"], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # Heads
        self.policy = nn.Sequential(
            nn.Linear(256, config["num_acts"]),
            nn.Softmax(dim=-1)
        )
        self.value = nn.Sequential(
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.hidden(x)
        dist = Categorical(probs=self.policy(x))
        val = self.value(x).reshape(-1)
        return dist, val

    def get_action(self, x_numpy):
        with torch.no_grad():
            x = torch.as_tensor(x_numpy, dtype=torch.float32)
            dist = self.get_dist(x)
            act = dist.sample()

        return act.cpu().numpy()

    def get_dist(self, x):
        x = self.hidden(x)
        dist = Categorical(probs=self.policy(x))
        return dist

    def get_value(self, x):
        x = self.hidden(x)
        val = self.value(x).reshape(-1)
        return val