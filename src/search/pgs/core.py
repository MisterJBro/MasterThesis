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

    def __init__(self, config, state, eval_channel, pol_head, val_head, idx, mcs, dyn_length, scale_vals, expl_entr, expl_kl, visit_counts, change_update, puct_c, trunc_len, pgs_lr, entr_c, kl_c, p_val):
        self.config = config
        self.eval_channel = eval_channel
        self.idx = idx
        self.num_players = config["num_players"]
        self.NodeClass = PGSNode
        self.expl_coeff = puct_c
        self.pgs_lr = pgs_lr
        self.entr_c = entr_c
        self.kl_c = kl_c
        self.p_val = p_val
        self.device = config["device"]
        self.trunc_len = trunc_len
        self.mcs = mcs
        self.dyn_length = dyn_length
        self.scale_vals = scale_vals
        self.expl_entr = expl_entr
        self.expl_kl = expl_kl
        self.visit_counts = visit_counts
        self.change_update = change_update

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
        if self.change_update:
            self.optim_pol = optim.Adam(self.sim_policy.parameters(), lr=self.pgs_lr)
        else:
            self.optim_pol = optim.SGD(self.sim_policy.parameters(), lr=self.pgs_lr)
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

            #qvals = self.root.get_action_values(self.config["num_acts"], default=self.root.val)
            #print(qvals.reshape(5,5).round(5))

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

        qvals = self.root.get_action_values(self.config["num_acts"], default=0)
        vals = -self.root.qvalue()

        if self.visit_counts:
            #print("Normal visit counts:")
            #print(self.root.get_normalized_visit_counts(self.config["num_acts"]).reshape(5,5).round(2))
            #print("QVALS:")
            #print(qvals.reshape(5,5).round(2))

            max_visits = np.max([child.num_visits for child in self.root.children])
            adv = qvals - self.root.val
            adv = adv / (np.abs(np.max(adv)) + 1e-8)
            pi = (100 + max_visits) * 0.005 * adv
        else:
            pi = self.root.get_normalized_visit_counts(self.config["num_acts"])

        visits = self.root.get_normalized_visit_counts(self.config["num_acts"]) * self.root.num_visits

        return {
            "q": qvals,
            "v": vals,
            "pi": pi,
            "visits": visits,
        }

    def expand(self, node):
        if node.state is None:
            node.create_state()
            pol_h, val_h = self.eval_fn(node.state.obs)
            node.pol_h = pol_h
            node.val_h = val_h
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
            return child
        else:
            return node

    def simulate(self, node):
        env = deepcopy(node.state.env)
        done = node.state.done
        obs = node.state.obs
        pol_h = node.pol_h
        val_h = node.val_h
        num_visits = node.num_visits

        # Option dynamic length
        if self.dyn_length:
            branching_factor = 7
            self.trunc_len = int(np.log(num_visits+1) / np.log(branching_factor)) + 1

        player, iter = 0, 0
        acts, rews, pol_hs, val_hs = [], [], [], []
        while not done:
            player = (player + 1) % self.num_players
            pol_hs.append(pol_h)
            val_hs.append(val_h)

            # Get action
            with torch.no_grad():
                logits = self.sim_policy(pol_h)
                logits = self.filter_actions(logits, legal_actions=[env.legal_actions()])
                act = Categorical(logits=logits).sample().item()
            acts.append(act)
            #print("Val: ", self.base_value(val_h.cpu()).item())
            #env.render()
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
            if iter >= self.trunc_len:
                break
            iter += 1

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
            last_val = None

        return {
            "num_visits": num_visits,
            "rew": np.array(rews),
            "last_val": last_val,
            "act": torch.tensor(acts).to(self.config["device"]),
            "pol_h": torch.concat(pol_hs, 0).to(self.config["device"]),
            "val_h":  torch.concat(val_hs, 0).to(self.config["device"]),
        }

    def train(self, traj):
        if len(traj) == 0:
            return 0

        # Get traj values
        num_visits = traj["num_visits"]
        rew = traj["rew"]
        last_val = traj["last_val"]
        act = traj["act"]
        pol_h_cpu = traj["pol_h"].clone().cpu()
        pol_h = traj["pol_h"].to(self.device)
        val_h = traj["val_h"].to(self.device)

        # Get return
        if self.num_players == 2:
            # Finished Game
            if last_val is None:
                ret = np.full(len(rew), -rew[-1])
                ret[::-2] = rew[-1]
            else:
                ret = np.full(len(rew), -last_val)
                ret[:-1][::-2] = last_val
        else:
            ret = discount_cumsum(rew, self.config["gamma"])[:-1]
        ret = torch.as_tensor(ret).to(self.device)
        adv = ret

        with torch.no_grad():
            val = self.base_value(val_h).reshape(-1)
            #val = np.concatenate((val.cpu().numpy(), [rew[-1]]))
        #adv = gen_adv_estimation(rew[:-1], val, self.config["gamma"], self.config["lam"])

        if self.change_update and self.num_players == 2 and last_val is not None:
            ret = val - val.mean()

        #adv = torch.as_tensor(adv).to(self.device)
        with torch.no_grad():
            base_dist = Categorical(logits=self.base_policy(pol_h_cpu).to(self.device))

        # MCS
        if self.mcs:
            val = val.cpu().numpy().reshape(-1)
            val[::2] = -val[::2]
            return val[-1]

        # REINFORCE
        #if self.iter < 20:
        self.optim_pol.zero_grad()
        dist = Categorical(logits=self.sim_policy(pol_h))
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
        #loss_value = F.huber_loss(self.sim_value(val_h).reshape(-1), ret)
        #loss_value.backward()
        #self.optim_val.zero_grad()

        # Calculate return
        if last_val is None:
            total_ret = -ret[0].numpy()
        else:
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
