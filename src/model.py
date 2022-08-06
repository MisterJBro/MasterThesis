import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import pathlib

PROJECT_PATH = pathlib.Path(__file__).parent.absolute().as_posix()


class ValueEquivalenceModel(nn.Module):
    """Value Equivalence Model."""

    def __init__(self, config):
        super(ValueEquivalenceModel, self).__init__()
        self.hidden_size = 128
        self.num_acts = config["num_acts"]

        # Representation function h
        self.rep = nn.Sequential(
            nn.Linear(config["flat_obs_dim"], 128),
            nn.ReLU(),
            nn.Linear(128, self.hidden_size*2),
        )

        # Dynamic function g
        self.dyn = nn.LSTM(self.num_acts, self.hidden_size, batch_first=True)

        # Prediction functions f, for policy, value and reward functions
        self.pre_rew = nn.Sequential(
            nn.Linear(self.hidden_size, 1),
        )
        self.pre_val = nn.Sequential(
            nn.Linear(self.hidden_size, 1),
        )
        self.pre_pi = nn.Sequential(
            nn.Linear(self.hidden_size, self.num_acts)
        )

        self.opt = optim.Adam(list(self.parameters()), lr=config["model_lr"])
        self.device = config["device"]
        self.to(self.device)

    def representation(self, obs):
        """Using the representation function, transform the given observation into a state representation."""
        s = self.rep(obs)
        s = s.unsqueeze(0)
        s = s.split(self.hidden_size, dim=-1)
        return s

    def dynamics(self, s, a_onehot):
        hidden, s_next = self.dyn(a_onehot, s)
        return hidden, s_next

    def get_reward(self, hidden):
        return self.pre_rew(hidden).reshape(-1)

    def get_value(self, hidden):
        return self.pre_val(hidden).reshape(-1)

    def get_policy(self, hidden):
        return Categorical(logits=self.pre_pi(hidden).reshape(-1, self.num_acts))

    def save(self, path=f'{PROJECT_PATH}/checkpoints/ve_model.pt'):
        torch.save({
            'parameters': self.state_dict(),
            'optimizer': self.opt.state_dict(),
        }, path)

    def load(self, path=f'{PROJECT_PATH}/checkpoints/ve_model.pt'):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['parameters'])
        self.opt.load_state_dict(checkpoint['optimizer'])