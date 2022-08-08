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
        self.hidden_size = 512
        self.num_layers = 2
        self.num_acts = config["num_acts"]

        # Representation function h
        self.rep = nn.Sequential(
            nn.Linear(config["flat_obs_dim"], 2*self.num_layers*self.hidden_size),
            #nn.Linear(config["flat_obs_dim"], self.hidden_size),
            #nn.ReLU(),
            #nn.Linear(self.hidden_size, 2*self.num_layers*self.hidden_size),
            #nn.Linear(self.hidden_size, self.hidden_size),
            #nn.ReLU(),
            #nn.ReLU(),
        )
        self.rep_rnn = nn.LSTM(config["flat_obs_dim"], self.hidden_size, batch_first=True)

        # Dynamic function g
        #self.num_acts
        self.dyn_linear = nn.Sequential(
            nn.Linear(self.num_acts, self.hidden_size),
            #nn.Unflatten(1, (1, -1)),
            #nn.Conv1d(in_channels=1, out_channels=int(self.hidden_size/self.num_acts), kernel_size=1),
            nn.ReLU(),
            #nn.Flatten(0, 1),
            #nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.dyn = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)

        # Prediction functions f, for policy, value and reward functions
        self.pre_rew = nn.Sequential(
            nn.Linear(self.hidden_size, 1),
        )
        self.pre_val = nn.Sequential(
            nn.Linear(self.hidden_size, 1),
        )
        self.pre_pi = nn.Sequential(
            nn.Linear(self.hidden_size, self.num_acts),
        )

        self.opt = optim.Adam(list(self.parameters()), lr=config["model_lr"])
        self.device = config["device"]
        self.to(self.device)

    def representation(self, obs):
        """Using the representation function, transform the given observation into a state representation."""
        s = self.rep(obs)
        s = s.reshape(s.shape[0], self.num_layers, -1)
        s = s.permute(1, 0, 2)
        s = s.split(self.hidden_size, dim=-1)
        return (s[0].contiguous(), s[1].contiguous())

    def dynamics(self, s, a_onehot):
        hidden, s_next = self.dyn(a_onehot, s)
        return hidden, s_next

    def get_reward(self, hidden):
        return self.pre_rew(hidden).reshape(-1)

    def get_value(self, hidden):
        return self.pre_val(hidden).reshape(-1)

    def get_policy(self, hidden):
        return Categorical(logits=self.pre_pi(hidden).reshape(-1, self.num_acts))

    def loss(self, data):
        pass

    def save(self, path=f'{PROJECT_PATH}/checkpoints/ve_model.pt'):
        torch.save({
            'parameters': self.state_dict(),
            'optimizer': self.opt.state_dict(),
        }, path)

    def load(self, path=f'{PROJECT_PATH}/checkpoints/ve_model.pt'):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['parameters'])
        self.opt.load_state_dict(checkpoint['optimizer'])