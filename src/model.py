import torch
import torch.nn as nn
import pathlib

PROJECT_PATH = pathlib.Path(__file__).parent.absolute().as_posix()


class ValueEquivalenceModel(nn.Module):
    """Value Equivalence Model."""

    def __init__(self, config):
        super(ValueEquivalenceModel, self).__init__()
        self.hidden_size = 128

        # Representation function h
        self.rep = nn.Sequential(
            nn.Linear(config["flat_obs_dim"], 128),
            nn.ReLU(),
            nn.Linear(128, self.hidden_size*2),
        )

        # Dynamic function g
        self.dyn = nn.LSTM(config["num_acts"], self.hidden_size, batch_first=True)

        # Prediction functions f, for policy, value and reward functions
        self.pre_pi = nn.Sequential(
            nn.Linear(self.hidden_size, config["num_acts"]),
            nn.Softmax(dim=-1)
        )
        self.pre_val = nn.Sequential(
            nn.Linear(self.hidden_size, 1),
        )
        self.pre_rew = nn.Sequential(
            nn.Linear(self.hidden_size, 1),
        )

    def representation(self, obs):
        """Using the representation function, transform the given observation into a state representation."""
        s = self.rep(obs)
        s = s.unsqueeze(0)
        s = s.split(self.hidden_size, dim=-1)
        return s

    def dynamics(self, s, a):
        # Convert action into one-hot encoded representation
        a_onehot = torch.zeros(a.shape[0], self.num_acts).to(a.device)
        a_onehot.scatter_(1, a.view(-1, 1), 1)
        a_onehot.unsqueeze_(1)

        hidden, s_next = self.dyn(a_onehot, s)
        hidden.squeeze_(1)
        return hidden, s_next

    def get_policy(self, hidden):
        return self.pre_pi(hidden)

    def get_value(self, hidden):
        return self.pre_val(hidden)

    def get_reward(self, hidden):
        return self.pre_rew(hidden)

    def save(self, path=f'{PROJECT_PATH}/checkpoints/ve_model.pt'):
        torch.save({
            'parameters': self.state_dict(),
        }, path)

    def load(self, path=f'{PROJECT_PATH}/checkpoints/ve_model.pt'):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['parameters'])