from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import torch.optim as optim
from src.search.node import PGSNode
from src.search.pgs.core import PGSCore
from src.search.state import ModelState
from src.train.processer import discount_cumsum, gen_adv_estimation
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence


class VEPGSCore(PGSCore):
    """ Core of Value Equivalent Policy Gradient Search. """

    def __init__(self, config, state, num_acts, eval_channel, pol_head, val_head, idx, mcs, dyn_length, scale_vals, expl_entr, expl_kl, visit_counts, change_update, puct_c, trunc_len, pgs_lr, entr_c, kl_c, p_val):
        super().__init__(config, state, eval_channel, pol_head, val_head, idx, mcs, dyn_length, scale_vals, expl_entr, expl_kl, visit_counts, change_update, puct_c, trunc_len, pgs_lr, entr_c, kl_c, p_val)
        self.num_acts = num_acts

    def reset(self, base_policy=None, base_value=None):
        # Reset simulation policy e.g. after each search
        del self.sim_policy
        del self.sim_value
        del self.optim_pol
        del self.optim_val

        if base_policy is not None:
            self.base_policy = base_policy
        if base_value is not None:
            self.base_value = base_value
        self.sim_policy = deepcopy(self.base_policy)
        self.sim_value = deepcopy(self.base_value)
        if self.change_update:
            self.optim_pol = optim.Adam(self.sim_policy.parameters(), lr=self.pgs_lr)
        else:
            self.optim_pol = optim.SGD(self.sim_policy.parameters(), lr=self.pgs_lr)
        self.optim_val = optim.Adam(self.sim_value.parameters(), lr=1e-3)

    def expand(self, node):
        if node.state is None:
            next_abs, rew, hidden = self.eval_abs(node.parent.state.abs, node.action)
            node.state = ModelState(next_abs, rew=rew)
            node.hidden = hidden
            return node
        if node == self.root:
            # Create new child nodes, lazy init
            actions = node.get_legal_actions()
            for action in actions:
                new_node = self.NodeClass(None, action=action, parent=node)
                node.children.append(new_node)

            # Create child with highest prior
            max_prior_indices = np.flatnonzero(node.priors == np.max(node.priors))
            if len(max_prior_indices) == 1:
                child = node.children[max_prior_indices[0]]
            else:
                child = node.children[np.random.choice(max_prior_indices)]

            next_abs, rew, hidden = self.eval_abs(child.parent.state.abs, child.action)
            child.state = ModelState(next_abs, rew=rew)
            child.hidden = hidden
            return child
        else:
            return node

    def simulate(self, node):
        abs = node.state.abs
        hidden = node.hidden
        num_visits = node.num_visits

        # Option dynamic length
        if self.dyn_length:
            branching_factor = 7
            self.trunc_len = int(np.log(num_visits+1) / np.log(branching_factor)) + 1

        player = 0
        acts, rews, hs = [], [], []
        for _ in range(self.trunc_len):
            player = (player + 1) % self.num_players
            hs.append(hidden)
            with torch.no_grad():
                act = Categorical(logits=self.sim_policy(hidden)).sample().item()
            acts.append(act)

            abs, rew, hidden = self.eval_abs(abs, act)

            # Add reward
            if self.num_players == 2:
                if player == 0:
                    rews.append(rew)
                else:
                    rews.append(-rew)
            else:
                rews.append(rew)

        if len(acts) == 0:
            return {}

        # Bootstrap last value
        with torch.no_grad():
            last_val = self.base_value(hidden).item()

        return {
            "num_visits": num_visits,
            "rew": np.array(rews),
            "last_val": last_val,
            "act": torch.as_tensor(np.array(acts)),
            "hidden": torch.concat(hs, 0),
        }

    def train(self, traj):
        if len(traj) == 0:
            return 0

        # Get traj values
        rew = traj["rew"]
        act = traj["act"]
        hidden = traj["hidden"].to(self.device)
        last_val = traj["last_val"]

        # Get return
        if self.num_players == 2:
            # Finished Game
            ret = np.full(len(rew), -last_val)
            ret[:-1][::-2] = last_val
        else:
            ret = discount_cumsum(rew, self.config["gamma"])[:-1]
        ret = torch.as_tensor(ret).to(self.device)

        # Value
        with torch.no_grad():
            val = self.base_value(hidden).reshape(-1)
            #val = np.concatenate((val.numpy(), [rew[-1]]))

        if self.change_update and self.num_players == 2:
            ret = val - val.mean()
        adv = ret

        #adv = gen_adv_estimation(rew[:-1], val, self.config["gamma"], self.config["lam"])
        #adv = torch.as_tensor(adv).to(self.device)
        with torch.no_grad():
            base_dist = Categorical(logits=self.base_policy(hidden))

        # MCS
        if self.mcs:
            val = val.cpu().numpy().reshape(-1)
            val[::2] = -val[::2]
            return val[-1]

        # REINFORCE
        #if self.iter < 20:
        self.optim_pol.zero_grad()
        dist = Categorical(logits=self.sim_policy(hidden))
        logp = dist.log_prob(act)
        loss_policy = -(logp * adv).mean()
        #print(dist.entropy().mean())
        loss = loss_policy
        if self.expl_entr:
            loss_entropy = -dist.entropy().mean()
            loss += self.entr_c*loss_entropy
        if self.expl_kl:
            loss_dist = kl_divergence(base_dist, dist).mean()
            loss += self.kl_c*loss_dist
        loss.backward()
        self.optim_pol.step()

        # Value function
        #self.optim_val.step()
        #loss_value = F.huber_loss(self.sim_value(hidden).reshape(-1), ret)
        #loss_value.backward()
        #self.optim_val.zero_grad()

        # Calculate return
        val = val.cpu().numpy().reshape(-1)
        val[::2] = -val[::2]

        if self.scale_vals:
            p = self.p_val
            k = len(val)
            log_dist = p**np.arange(k)/(-k*np.log(1-p))
            log_dist = log_dist / np.sum(log_dist)
            total_ret = val @ log_dist
        else:
            total_ret = val[-1]

        return total_ret

    def set_root(self, state):
        self.iter = 0
        self.root = PGSNode(state)
        if state is not None:
            prob, abs, val = self.eval_obs(self.root)
            self.root.priors = prob[self.root.get_legal_actions()]
            self.root.val = val
            self.root.state.abs = abs

    def eval_obs(self, node):
        self.eval_channel.send({
            "obs": node.state.obs,
            "ind": self.idx,
        })
        msg = self.eval_channel.recv()
        return msg["prob"], msg["abs"], msg["val"]

    def eval_abs(self, abs, act):
        self.eval_channel.send({
            "abs": abs,
            "act": act,
            "ind": self.idx,
        })
        msg = self.eval_channel.recv()
        return msg["abs"], msg["rew"][0], msg["hidden"]