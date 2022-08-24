import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from torch.distributions import Categorical

from multiprocessing import Process, Pipe
from src.search.node import PUCTNode
from src.search.tree import Tree
from src.search.evaluator import EvaluatorPGS
from src.train.processer import discount_cumsum


class PGSTree(Tree):
    """ Small Tree for discrete Policy Gradient search. Only depth of one. """

    def __init__(self, state, eval_channel, pol_head, val_head, config, idx=0):
        self.config = config
        self.eval_channel = eval_channel
        self.idx = idx
        self.num_players = config["num_players"]
        self.NodeClass = PUCTNode
        self.expl_coeff = config["puct_c"]
        self.device = config["device"]
        self.trunc_len = config["pgs_trunc_len"]

        self.sim_policy = pol_head
        self.value = val_head
        self.optim = optim.Adam(self.sim_policy.parameters(), lr=config["pgs_lr"])
        self.set_root(state)

    def search(self, iters=1_000):
        iter = 0

        while iter < iters:
            leaf = self.select()
            new_leaf = self.expand(leaf)
            rew, pol_hs, val_hs, act = self.simulate(new_leaf)
            ret = self.update_policy(rew, pol_hs, val_hs, act)
            self.backpropagate(new_leaf, ret)
            iter += 1

        return self.root.get_action_values()

    def expand(self, node):
        if node.state is None:
            node.create_state()
            return node
        if node.state.is_terminal():
            return node
        if node == self.root:
            # Create new child nodes, lazy init
            actions = node.state.get_possible_actions()
            for action in actions:
                new_node = self.NodeClass(None, action=action, parent=node)
                node.children.append(new_node)

            # Pick child node
            child = np.random.choice(node.children)
            child.create_state()
            return child
        else:
            return node

    def simulate(self, node):
        env = deepcopy(node.state.env)
        done = node.state.done
        obs = node.state.obs

        player, iter = 0, 0
        acts, rews, pol_hs, val_hs = [], [], [], []
        while not done:
            player = (player + 1) % self.num_players
            pol_h, val_h = self.eval_fn(obs)
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

        # Bootstrap last value
        if iter >= self.trunc_len:
            _, val_h = self.eval_fn(obs)
            with torch.no_grad():
                rews.append(self.value(val_h).item())
        else:
            rews.append(0)

        if len(acts) == 0:
            return rews, pol_hs, val_hs, acts

        return np.array(rews), torch.stack(pol_hs, 0), torch.stack(val_hs, 0), torch.tensor(acts)

    def update_policy(self, rew, pol_hs, val_hs, act):
        if len(act) == 0:
            return 0

        ret = discount_cumsum(rew, self.config["gamma"])
        total_ret = ret[0]
        ret = torch.as_tensor(ret).to(self.device)
        adv = ret[:-1]
        pol_hs = pol_hs.to(self.device)
        val_hs = val_hs.to(self.device)

        # REINFORCE
        dist = Categorical(logits=self.sim_policy(pol_hs))
        logp = dist.log_prob(act)
        loss_policy = -(logp * adv).mean()
        loss_entropy = - dist.entropy().mean()
        loss = loss_policy + 0.01 * loss_entropy

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.sim_policy.parameters(), self.config["grad_clip"])
        self.optim.step()

        return total_ret

    def set_root(self, state):
        self.root = PUCTNode(state)
        if state is not None:
            pol_h, _ = self.eval_fn(self.root.state.obs)
            with torch.no_grad():
                probs = F.softmax(self.sim_policy(pol_h), dim=-1)
            self.root.priors = probs.cpu().numpy()

    def eval_fn(self, obs):
        self.eval_channel.send({
            "obs": obs,
            "ind": self.idx,
        })
        msg = self.eval_channel.recv()
        pol_hidden = msg["pol_hidden"]
        val_hidden = msg["val_hidden"]
        return pol_hidden, val_hidden

class PGSTreeWorker(Process, PGSTree):
    """ Multiprocessing Tree Worker, for parallelization of MCTS."""
    def __init__(self, iters, eval_channel,  pol_head, val_head, idx, config, channel):
        PGSTree.__init__(self, None, eval_channel, pol_head, val_head, config, idx=idx)
        Process.__init__(self)

        self.iters = iters
        self.channel = channel

    def run(self):
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

    def search(self, state, iters=None):
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
        for w in self.workers:
            w.join()
