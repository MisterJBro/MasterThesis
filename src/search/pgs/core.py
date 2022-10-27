from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import torch.optim as optim
from src.search.mcts.core import MCTSCore
from src.search.node import PGSNode
from src.train.processer import discount_cumsum, gen_adv_estimation
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence


class PGSCore(MCTSCore):
    """ Small Tree for discrete Policy Gradient search. Only depth of one. """

    def __init__(self, config, state, eval_channel, pol_head, val_head, idx=0):
        self.config = config
        self.eval_channel = eval_channel
        self.idx = idx
        self.num_players = config["num_players"]
        self.NodeClass = PGSNode
        self.expl_coeff = config["puct_c"]
        self.device = config["device"]
        self.trunc_len = config["pgs_trunc_len"]

        self.sim_policy = None
        self.sim_value = None
        self.optim_pol = None
        self.optim_val = None
        self.reset(base_policy=pol_head, base_value=val_head)
        self.set_root(state)

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
        self.sim_policy = deepcopy(self.base_policy.cpu()).to(self.config["device"])
        self.sim_value = deepcopy(self.base_value.cpu()).to(self.config["device"])
        self.optim_pol = optim.Adam(self.sim_policy.parameters(), lr=self.config["pgs_lr"])
        self.optim_val = optim.Adam(self.sim_value.parameters(), lr=1e-3)

    def search(self, iters):
        rets = []
        while self.iter < iters:
            leaf = self.select()
            new_leaf = self.expand(leaf)
            traj = self.simulate(new_leaf)
            ret = self.train(traj)
            self.backpropagate(new_leaf, ret)
            rets.append(ret)
            self.iter += 1

            qvals = self.root.get_action_values(self.config["num_acts"], default=self.root.val)
            print("QVALS:")
            print(qvals.reshape(5,5).round(2))

        def plot():
            import os
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
            #print([float('{:.2f}'.format(x)) for x in rets])
            sns.set_theme()
            x = np.arange(len(rets))
            poly = np.polyfit(x, rets, 1)
            poly_y = np.poly1d(poly)(x)
            plt.plot(x, rets, alpha=0.5)
            plt.plot(x, poly_y)
            plt.show()
        #plot()

        if self.config["search_return_adv"]:
            qvals = self.root.get_action_values(self.config["num_acts"], default=self.root.val)
            print("Normal visit counts:")
            print(self.root.get_normalized_visit_counts(self.config["num_acts"]).reshape(5,5).round(2))
            print("QVALS:")
            qvals[qvals < -100] = 0
            print(qvals.reshape(5,5).round(2))

            max_visits = np.max([child.num_visits for child in self.root.children])
            adv = qvals - self.root.val
            adv = adv / (np.abs(np.max(adv)) + 1e-8)
            return (100 + max_visits) * 0.005 * adv
        return self.get_normalized_visit_counts(self.config["num_acts"])

    def expand(self, node):
        if node.state is None:
            node.create_state()
            pol_h, val_h = self.eval_fn(node.state.obs)
            node.pol_h = pol_h
            node.val_h = val_h
            #with torch.no_grad():
            #    node.qval = node.state.rew + self.config["gamma"] * self.base_value(val_h).item()
            return node
        if node.is_terminal():
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

            child.create_state()
            pol_h, val_h = self.eval_fn(child.state.obs)
            child.pol_h = pol_h
            child.val_h = val_h
            #with torch.no_grad():
            #    child.qval = child.state.rew + self.config["gamma"] * self.base_value(val_h).item()
            return child
        else:
            return node

    def simulate(self, node):
        env = deepcopy(node.state.env)
        done = node.state.done
        obs = node.state.obs
        pol_h = node.pol_h
        val_h = node.val_h

        player, iter = 0, 0
        acts, rews, pol_hs, val_hs = [], [], [], []
        while not done:
            player = (player + 1) % self.num_players
            pol_hs.append(pol_h)
            val_hs.append(val_h)

            with torch.no_grad():
                logits = self.sim_policy(pol_h)
                logits = self.filter_actions(logits, legal_actions=[env.available_actions()])
                act = Categorical(logits=logits).sample().item()
            acts.append(act)
            env.render()
            obs, rew, done, _ = env.step(act)

            # Add reward
            if self.num_players == 2:
                if player == 0:
                    rews.append(rew)
                else:
                    rews.append(-rew)
            else:
                rews.append(rew)

            # Truncated rollout
            iter += 1
            if iter >= self.trunc_len:
                break

            # Get next hidden states
            pol_h, val_h = self.eval_fn(obs)

        if len(acts) == 0:
            return {}

        # Bootstrap last value
        if iter >= self.trunc_len:
            _, val_h = self.eval_fn(obs)
            with torch.no_grad():
                last_val = self.base_value(val_h.cpu()).item()
        else:
             last_val = 0
        rews.append(last_val)
        rews = np.array(rews)

        return {
            "rew": rews,
            "act": torch.tensor(acts).to(self.config["device"]),
            "pol_h": torch.concat(pol_hs, 0).to(self.config["device"]),
            "val_h":  torch.concat(val_hs, 0).to(self.config["device"]),
        }

    def train(self, traj):
        if len(traj) == 0:
            return 0

        # Get traj values
        rew = traj["rew"]
        act = traj["act"]
        pol_h_cpu = traj["pol_h"].clone().cpu()
        pol_h = traj["pol_h"].to(self.device)
        val_h = traj["val_h"].to(self.device)

        # Get ret, vals and adv
        ret = discount_cumsum(rew, self.config["gamma"])[:-1]
        total_ret = ret[0]
        ret = torch.as_tensor(ret).to(self.device)

        with torch.no_grad():
            val = self.sim_value(val_h).reshape(-1)
            val = np.concatenate((val.cpu().numpy(), [rew[-1]]))
        adv = gen_adv_estimation(rew[:-1], val, self.config["gamma"], self.config["lam"])
        adv = torch.as_tensor(adv).to(self.device)
        with torch.no_grad():
            base_dist = Categorical(logits=self.base_policy(pol_h_cpu).to(self.device))

        # REINFORCE
        #if self.iter < 20:
        self.optim_pol.zero_grad()
        dist = Categorical(logits=self.sim_policy(pol_h))
        logp = dist.log_prob(act)
        loss_policy = -(logp * adv).mean()
        #print(dist.entropy().mean())
        loss_dist = kl_divergence(base_dist, dist).mean()
        #loss_entropy = -dist.entropy().mean()
        loss = loss_policy# + 0.1 * loss_dist
        loss.backward()
        self.optim_pol.step()

        # Value function
        self.optim_val.step()
        loss_value = F.huber_loss(self.sim_value(val_h).reshape(-1), ret)
        loss_value.backward()
        self.optim_val.zero_grad()

        return total_ret

    def set_root(self, state):
        self.iter = 0
        self.root = PGSNode(state)
        if state is not None:
            pol_h, val_h = self.eval_fn(self.root.state.obs)
            with torch.no_grad():
                logits = self.base_policy(pol_h.cpu())
                prob = F.softmax(logits, dim=-1)
                val = self.base_value(val_h.cpu())
            self.logits = logits.cpu().numpy().reshape(-1)

            self.root.priors = prob.cpu().numpy().reshape(-1)[self.root.get_legal_actions()]
            self.root.val = val.cpu().item()
            self.root.pol_h = pol_h
            self.root.val_h = val_h

    def filter_actions(self, logits, legal_actions=None):
        if legal_actions is None:
            return logits

        # Mask out invalid actions
        MASK_VALUE = -10e8 if logits.dtype == torch.float32 else -1e4
        new_logits = torch.full(logits.shape, MASK_VALUE, dtype=logits.dtype).to(self.device)
        for i, row in enumerate(legal_actions):
            new_logits[i, row] = logits[i, row]
        return new_logits

    def eval_fn(self, obs):
        self.eval_channel.send({
            "obs": obs,
            "ind": self.idx,
        })
        msg = self.eval_channel.recv()
        return msg["pol_h"], msg["val_h"]
