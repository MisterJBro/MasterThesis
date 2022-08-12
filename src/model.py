from ensurepip import bootstrap
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import pathlib

PROJECT_PATH = pathlib.Path(__file__).parent.absolute().as_posix()


class ValueEquivalenceModel(nn.Module):
    """Value Equivalence Model."""

    def __init__(self, config):
        super(ValueEquivalenceModel, self).__init__()
        self.config = config
        self.hidden_size = 512
        self.num_layers = 2
        self.num_acts = config["num_acts"]

        # Representation function h
        self.rep = nn.Sequential(
            nn.Linear(config["flat_obs_dim"], 2*self.num_layers*self.hidden_size),
        )
        self.rep_rnn = nn.LSTM(config["flat_obs_dim"], self.hidden_size, batch_first=True)

        # Dynamic function g
        self.dyn_linear = nn.Sequential(
            nn.Linear(self.num_acts, self.hidden_size),
            nn.ReLU(),
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
        obs = data["obs"]
        act = data["act"]
        rew = data["rew"]
        done = data["done"]
        ret = data["ret"]
        logits = data["logits"]
        last_val = data["last_val"]
        sections = data["sections"]
        scalar_loss = nn.HuberLoss()
        act_onehot = to_onehot(act, self.num_acts)

        # Prepare episodes
        episodes = []
        num_bootstraps = 0
        for (start, end) in sections:
            act_ep = []
            lengths = []
            rew_targets = []
            val_targets = []
            dist_targets = []
            bootstrap = done[end-1]
            if bootstrap:
                next_val_targets = torch.concat([ret[start+1:end], torch.zeros(1, device=self.device)])
            else:
                next_val_targets = torch.concat([ret[start+1:end], last_val[num_bootstraps].unsqueeze(0)])
                num_bootstraps += 1

            for t in range(start, end):
                end_ep = min(t+self.config["model_unroll_len"], end)
                cut = (end_ep - t) != self.config["model_unroll_len"]

                next_actions = act_onehot[t:end_ep]
                rew_target = rew[t:end_ep]
                val_target = next_val_targets[t-start:end_ep-start]
                dist_target = logits[t:end_ep]
                if bootstrap and cut:
                    rew_target = torch.concat([rew_target, torch.zeros(1, device=self.device)])
                    val_target = torch.concat([val_target, torch.zeros(1, device=self.device)])
                    random_action = torch.randint(0, self.num_acts, (1,), device=self.device)
                    random_action = to_onehot(random_action, self.num_acts)
                    next_actions = torch.concat([next_actions, random_action])
                    dist_target = torch.concat([dist_target, logits[end_ep-1].unsqueeze(0)])

                rew_targets.append(rew_target)
                val_targets.append(val_target)
                dist_targets.append(dist_target)
                act_ep.append(next_actions)
                lengths.append(next_actions.shape[0])

            act_ep = torch.concat(act_ep)
            rew_targets = torch.concat(rew_targets)
            val_targets = torch.concat(val_targets)
            dist_targets = torch.concat(dist_targets)
            dist_targets = Categorical(logits=dist_targets)

            episodes.append({
                'start': start,
                'end': end,
                'act_ep': act_ep,
                'lengths': lengths,
                'rew_targets': rew_targets,
                'val_targets': val_targets,
                'dist_targets': dist_targets,
            })
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
                lengths = ep["lengths"]
                rew_targets = ep["rew_targets"]
                val_targets = ep["val_targets"]
                dist_targets = ep["dist_targets"]

                act_ep = self.dyn_linear(act_ep)
                tmp = []
                tmp_index = 0
                for l in lengths:
                    tmp_act = act_ep[tmp_index:tmp_index + l]
                    tmp.append(tmp_act)
                    tmp_index += l
                act_ep = tmp
                act_ep = pad_sequence(act_ep, batch_first=True)
                act_ep = pack_padded_sequence(act_ep, lengths=lengths, batch_first=True)

                state = self.representation(obs[start:end])
                hidden, _ = self.dynamics(state, act_ep)
                hidden, lengths = pad_packed_sequence(hidden, batch_first=True)
                hidden = [hidden[i][:len] for i, len in enumerate(lengths)]
                hidden = torch.concat(hidden)

                model_rew = self.get_reward(hidden)
                model_val = self.get_value(hidden)
                model_dist = self.get_policy(hidden)

                loss_rew = scalar_loss(model_rew, rew_targets)
                loss_val = scalar_loss(model_val, val_targets)
                # Mode seeking KL
                loss_dist = kl_divergence(dist_targets, model_dist).mean()
                loss = loss_rew + loss_val + loss_dist
                losses.append(loss)

                # Minibatch update
                steps += end - start
                if steps >= batch_size:
                    self.opt.zero_grad()
                    loss = torch.mean(torch.stack(losses))
                    loss.backward()
                    self.opt.step()
                    steps = 0
                    losses = []

            if len(losses) != 0:
                self.opt.zero_grad()
                loss = torch.mean(torch.stack(losses))
                loss.backward()
                self.opt.step()

    def save(self, path=f'{PROJECT_PATH}/checkpoints/ve_model.pt'):
        torch.save({
            'parameters': self.state_dict(),
            'optimizer': self.opt.state_dict(),
        }, path)

    def load(self, path=f'{PROJECT_PATH}/checkpoints/ve_model.pt'):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['parameters'])
        self.opt.load_state_dict(checkpoint['optimizer'])

def to_onehot(a, num_acts):
    # Convert action into one-hot encoded representation
    a_onehot = torch.zeros(a.shape[0], num_acts).to(a.device)
    a_onehot.scatter_(1, a.view(-1, 1), 1)

    return a_onehot