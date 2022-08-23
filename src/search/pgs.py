import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from multiprocessing import Process
from src.search.node import UCTNode, PUCTNode, DirichletNode
from src.search.tree import Tree
from src.search.util import measure_time


class MCSTree(PGSTree):
    """ Monte Carlo Search. Using PGS without any update of simulation policy. """

    def update_policy(self, rew, hs):
        pass


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

        self.sim_policy = pol_head
        self.value = val_head
        self.optim = optim.Adam(self.sim_policy.parameters(), lr=1e-4)
        self.set_root(state)

    def search(self, iters=1_000):
        iter = 0

        while iter < iters:
            leaf = self.select()
            new_leaf = self.expand(leaf)
            rew, hs = self.simulate(new_leaf)
            self.update_policy(rew, hs)
            self.backpropagate(new_leaf, rew, hs)
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
        env = deepcopy(self.state.env)
        done = self.state.done
        obs = self.state.obs

        player, iter = 0, 0
        rews, hs = [], []
        while not done:
            player = (player + 1) % num_players
            hidden = self.eval_fn(obs)
            hs.append(hidden)
            act = Categorical(logits=self.sim_policy(hidden)).sample()
            obs, rew, done, _ = env.step(act)

            # Add return
            if num_players == 2:
                if player == 0:
                    rews.append(rew)
                else:
                    rews.append(-rew)
            else:
                rews.append(rew)

            # Truncated rollout
            iter += 1
            if iter >= 10:
                hidden = self.eval_fn(obs)
                hs.append(hidden)
                rews.append(self.value(hidden))
                break
        return np.array(rews), torch.concat(hs, 0)

    def update_policy(self, rew, hs):
        ret = discount_cumsum(rew, self.config["gamma"])
        ret = torch.tensor(ret).to(self.device)
        adv = ret
        hs = hs.to(self.device)

        # REINFORCE
        dist = Categorical(logits=self.sim_policy(hs))
        logp = dist.log_prob(act)
        loss_policy = -(logp * adv).mean()
        loss_entropy = - dist.entropy().mean()
        loss = loss_policy + 0.01 * loss_entropy

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.sim_policy.parameters(),  self.config["grad_clip"])
        self.optim.step()

    def set_root(self, state):
        self.root = PUCTNode(state)
        if state is not None:
            probs, _ = self.eval_fn(self.root.state.obs)
            self.root.priors = probs

    def eval_fn(self, obs):
        self.eval_channel.send({
            "obs": obs,
            "ind": self.idx,
        })
        hidden = self.eval_channel.recv()
        return hidden

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