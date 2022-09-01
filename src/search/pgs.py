import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence

from multiprocessing import Process, Pipe
from src.search.node import PUCTNode, PGSNode
from src.search.tree import Tree
from src.search.evaluator import EvaluatorPGS
from src.train.processer import discount_cumsum, gen_adv_estimation


class PGSTree(Tree):
    """ Small Tree for discrete Policy Gradient search. Only depth of one. """

    def __init__(self, state, eval_channel, pol_head, val_head, config, idx=0):
        self.config = config
        self.eval_channel = eval_channel
        self.idx = idx
        self.num_players = config["num_players"]
        self.NodeClass = PGSNode
        self.expl_coeff = config["puct_c"]
        self.device = config["device"]
        self.trunc_len = config["pgs_trunc_len"]

        self.reset_policy(base_policy=pol_head, base_value=val_head)
        self.set_root(state)

    def reset_policy(self, base_policy=None, base_value=None):
        if base_policy is not None:
            self.base_policy = base_policy
        if base_value is not None:
            self.base_value = base_value
        self.sim_policy = deepcopy(self.base_policy)
        self.value = deepcopy(self.base_value)
        self.optim_pol = optim.Adam(self.sim_policy.parameters(), lr=self.config["pgs_lr"])
        self.optim_val = optim.Adam(self.value.parameters(), lr=self.config["pgs_lr"])

    def search(self, iters=1_000):
        rets = []
        while self.iter < iters:
            leaf = self.select()
            new_leaf = self.expand(leaf)
            rew, pol_hs, val_hs, act = self.simulate(new_leaf)
            ret = self.update_policy(rew, pol_hs, val_hs, act)
            self.backpropagate(new_leaf, ret)
            rets.append(ret)
            self.iter += 1

        def plot():
            import os
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
            print([float('{:.2f}'.format(x)) for x in rets])
            sns.set_theme()
            x = np.arange(len(rets))
            poly = np.polyfit(x, rets, 1)
            poly_y = np.poly1d(poly)(x)
            plt.plot(x, rets, alpha=0.5)
            plt.plot(x, poly_y)
            plt.show()
        #plot()
        if self.config["tree_output_qvals"]:
            #print(self.logits, self.root.get_action_values())
            return self.logits + self.root.get_action_values()
            #return self.root.get_action_values()
        return self.get_normalized_visit_counts()

    def expand(self, node):
        if node.state is None:
            node.create_state()
            pol_h, val_h = self.eval_fn(node.state.obs)
            node.pol_h = pol_h
            node.val_h = val_h
            return node
        if node.state.is_terminal():
            return node
        if node == self.root:
            # Create new child nodes, lazy init
            actions = node.state.get_possible_actions()
            for action in actions:
                new_node = self.NodeClass(None, action=action, parent=node)
                node.children.append(new_node)

            # Create child with highest prior
            max_prior_indices = np.flatnonzero(node.priors == np.max(node.priors))
            if len(max_prior_indices) > 1:
                child = node.children[np.random.choice(max_prior_indices)]
            else:
                child = node.children[max_prior_indices[0]]

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

        player, iter = 0, 0
        acts, rews, pol_hs, val_hs = [], [], [], []
        while not done:
            player = (player + 1) % self.num_players
            pol_hs.append(pol_h)
            val_hs.append(val_h)

            with torch.no_grad():
                act = Categorical(logits=self.sim_policy(pol_h)).sample().item()
            acts.append(act)
            obs, rew, done, _ = env.step(act)

            # Add return
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

        # Bootstrap last value
        if iter >= self.trunc_len:
            _, val_h = self.eval_fn(obs)
            with torch.no_grad():
                rews.append(self.base_value(val_h).item())
        else:
            rews.append(0)

        if len(acts) == 0:
            return rews, pol_hs, val_hs, acts

        return np.array(rews), torch.stack(pol_hs, 0), torch.stack(val_hs, 0), torch.tensor(acts)

    def update_policy(self, rew, pol_hs, val_hs, act):
        if len(act) == 0:
            return 0

        ret = discount_cumsum(rew, self.config["gamma"])[:-1]
        total_ret = ret[0]
        ret = torch.as_tensor(ret).to(self.device)
        pol_hs = pol_hs.to(self.device)
        val_hs = val_hs.to(self.device)
        with torch.no_grad():
            val = self.value(val_hs).numpy().reshape(-1)
        adv = gen_adv_estimation(rew[:-1], np.concatenate((val, [rew[-1]])), self.config["gamma"], 0.97)
        adv = torch.as_tensor(adv).to(self.device)
        with torch.no_grad():
            base_dist = Categorical(logits=self.base_policy(pol_hs))
        #adv = ret - val
        #print(adv-adv2)
        #quit()

        # REINFORCE
        self.optim_pol.zero_grad()
        dist = Categorical(logits=self.sim_policy(pol_hs))
        logp = dist.log_prob(act)
        loss_policy = -(logp * adv).mean()
        loss_dist = kl_divergence(base_dist, dist).mean()
        loss_entropy = -dist.entropy().mean()
        loss = loss_policy #+ 0.1 * loss_dist #+ 0.1 * loss_entropy

        loss.backward()
        nn.utils.clip_grad_norm_(self.sim_policy.parameters(), self.config["grad_clip"])
        self.optim_pol.step()

        # Value function
        self.optim_val.zero_grad()
        loss_value = F.huber_loss(self.value(val_hs).reshape(-1), ret)
        loss_value.backward()
        nn.utils.clip_grad_norm_(self.value.parameters(), self.config["grad_clip"])
        self.optim_val.step()

        return total_ret

    def set_root(self, state):
        self.reset_policy()
        self.iter = 0
        self.root = PGSNode(state)
        if state is not None:
            pol_h, _ = self.eval_fn(self.root.state.obs)
            with torch.no_grad():
                logits = self.sim_policy(pol_h)
                probs = F.softmax(logits, dim=-1)
            self.logits = logits.numpy().reshape(-1)
            self.root.priors = probs.cpu().numpy()

    def eval_fn(self, obs):
        self.eval_channel.send({
            "obs": obs,
            "ind": self.idx,
        })
        msg = self.eval_channel.recv()
        pol_h = msg["pol_h"]
        val_h = msg["val_h"]
        return pol_h, val_h


class PGSTreeWorker(Process, PGSTree):
    """ Multiprocessing Tree Worker, for parallelization of MCTS."""
    def __init__(self, iters, eval_channel,  pol_head, val_head, idx, config, channel):
        Process.__init__(self)
        PGSTree.__init__(self, None, eval_channel, pol_head, val_head, config, idx=idx)

        self.iters = iters
        self.channel = channel

    def run(self):
        self.reset_policy()

        msg = self.channel.recv()
        while msg["command"] != "close":
            if msg["command"] == "search":
                self.set_root(msg["state"])
                if msg["iters"] is not None:
                    iters = msg["iters"]
                else:
                    iters = self.iters
                qvals = self.search(iters)
                self.channel.send(qvals)
            elif msg["command"] == "update":
                self.reset_policy(base_policy=msg["pol_head"], base_value=msg["val_head"])

            msg = self.channel.recv()


class PGS:
    """ Policy Gradient Search with PUCT."""

    def __init__(self, policy, config):
        self.config = config
        self.num_workers = config["num_trees"]
        self.num_iters = config["pgs_iters"]

        # Create parallel tree workers
        pipes = [Pipe() for _ in range(self.num_workers)]
        eval_pipes = [Pipe() for _ in range(self.num_workers)]
        eval_master_pipe = Pipe()
        self.channels = [p[0] for p in pipes]
        self.eval_channel = eval_master_pipe[0]
        self.num_iters_worker = int(self.num_iters/self.num_workers)
        self.rest_iters = (self.num_iters % self.num_workers) + self.num_iters_worker

        self.workers = []
        for i in range(self.num_workers):
            iters = self.rest_iters if i == self.num_workers-1 else self.num_iters_worker
            eval_pipe = eval_pipes[i][1]

            pol_head = deepcopy(policy.policy_head)
            val_head = deepcopy(policy.value_head)
            worker = PGSTreeWorker(iters, eval_pipe, pol_head, val_head, i, config, pipes[i][1])
            worker.start()
            self.workers.append(worker)

        # Create evaluation worker
        eval_channels = [p[0] for p in eval_pipes]
        eval_master_channel = eval_master_pipe[1]
        self.eval_worker = EvaluatorPGS(policy, eval_channels, eval_master_channel, device=config["device"], batch_size=config["pgs_eval_batch"], timeout=config["pgs_eval_timeout"])
        self.eval_worker.start()

    def update_policy(self, state_dict):
        self.eval_channel.send({
            "command": "update",
            "state_dict": state_dict,
        })
        pol_head, val_head = self.eval_channel.recv()
        for c in self.channels:
            c.send({
                "command": "update",
                "pol_head": deepcopy(pol_head),
                "val_head": deepcopy(val_head),
            })

    def search(self, state, iters=None):
        self.eval_channel.send({"command": "clear cache"})
        for c in self.channels:
            c.send({
                "command": "search",
                "state": deepcopy(state),
                "iters": iters,
            })
        msg = np.stack([c.recv() for c in self.channels])

        qvals = np.mean(msg, axis=0)
        return qvals

    def distributed_search(self, states):
        self.eval_channel.send({"command": "clear cache"})
        i = 0
        dists = []
        len_states = len(states)
        while i < len_states:
            max_c_idx = self.num_workers
            for c_idx, c in enumerate(self.channels):
                c.send({
                    "command": "search",
                    "state": states[i],
                    "iters": self.num_iters,
                })
                i += 1
                if i >= len_states:
                    max_c_idx = c_idx+1
                    break
            msg = [c.recv() for c in self.channels[:max_c_idx]]
            dists.extend(msg)
        return np.array(dists)

    def close(self):
        for c in self.channels + [self.eval_channel]:
            c.send({"command": "close"})
        for w in self.workers + [self.eval_worker]:
            w.join()
