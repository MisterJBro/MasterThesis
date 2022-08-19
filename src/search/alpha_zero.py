import numpy as np
import torch
from numba import jit

import numpy as np
from copy import deepcopy
from util import measure_time
from multiprocessing import Pipe, freeze_support, Process


class AlphaZero:
    """ Monte Carlo Tree Search, with root parallelization."""
    def __init__(self, config):
        self.config = config
        self.num_workers = config["num_trees"]
        self.num_iters = config["mcts_iters"]

        # Create parallel tree workers
        pipes = [Pipe() for _ in range(self.num_workers)]
        self.channels = [p[0] for p in pipes]
        self.num_iters_worker = int(self.num_iters/self.num_workers)
        self.rest_iters = (self.num_iters % self.num_workers) + self.num_iters_worker
        self.workers = [
            TreeWorker(self.rest_iters if i == self.num_workers-1 else self.num_iters_worker, config, pipes[i][1])
            for i in range(self.num_workers)
        ]
        for w in self.workers:
            w.start()

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

    def close(self):
        for c in self.channels:
            c.send({"command": "close"})
        for w in self.workers:
            w.join()

class Evaluator:#(Process)
    """Evaluation Service for Nodes."""
    def __init__(self, policy):
        self.policy = policy

    def __call__(self, node):
        obs = torch.as_tensor(node.state.obs)
        with torch.no_grad():
            dist, val = self.policy(obs)
        probs = dist.probs.cpu().numpy()
        val = val.cpu().item()
        return probs, val

class AZTree:
    """ Search Tree presentation. """

    def __init__(self, state, policy, config):
        self.config = config
        self.eval = Evaluator(policy)
        self.num_players = config["num_players"]
        self.NodeClass = PUCTNode
        self.set_root(state)

    def search(self, iters=1_000):
        iter = 0
        select_time = np.zeros(1)
        expand_time = np.zeros(1)
        simulate_time = np.zeros(1)
        backpropagate_time = np.zeros(1)

        while iter < iters:
            leaf = measure_time(lambda: self.select(), select_time)
            new_leaf = measure_time(lambda: self.expand(leaf), expand_time)
            ret = measure_time(lambda: self.simulate(new_leaf), simulate_time)
            measure_time(lambda: self.backpropagate(new_leaf, ret), backpropagate_time)
            iter += 1

        #print(f"Select time: {select_time.item():0.2f}s")
        #print(f"Expand time: {expand_time.item():0.2f}s")
        #print(f"Simulate time: {simulate_time.item():0.2f}s")
        #print(f"Backpropagate time: {backpropagate_time.item():0.2f}s \n")

        return self.get_policy_targets()

    def select(self):
        node = self.root
        while not node.is_leaf():
            node = node.select_child(self.config["puct_c"])
        return node

    def expand(self, node):
        if node.state is None:
            node.create_state()
            return node
        if node.state.is_terminal():
            return node

        # Create new child nodes, lazy init
        actions = node.state.get_possible_actions()
        for action in actions:
            new_node = self.NodeClass(None, action=action, parent=node)
            node.children.append(new_node)

        # Pick child node
        child = np.random.choice(node.children)
        child.create_state()
        return child

    def simulate(self, node):
        probs, val = self.eval(node)
        node.priors = probs
        return val

    def backpropagate(self, node, ret):
        curr_ret = ret

        while node is not None:
            node.num_visits += 1
            node.total_rews += curr_ret

            flip = -1.0 if self.num_players == 2 else 1.0
            curr_ret = flip * (node.state.rew + curr_ret)
            node = node.parent

    def set_root(self, state):
        self.root = DirichletNode(state)
        probs, _ = self.eval(self.root)
        self.root.priors = probs

    def get_policy_targets(self, temp=1.0):
        return [child.num_visits ** (1/temp) / self.root.num_visits ** (1/temp) for child in self.root.children]

class TreeWorker(Process, AZTree):
    """ Multiprocessing Tree Worker, for parallelization of MCTS."""
    def __init__(self, iters, config, channel):
        AZTree.__init__(self, None, config)
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

class PUCTNode:
    """Predictor + UCB for Trees"""
    def __init__(self, state, action=None, parent=None):
        self.num_visits = 0
        self.total_rews = 0.0
        self.priors = None

        self.state = state
        self.action = action
        self.parent = parent
        if parent is not None:
            self.depth = parent.depth + 1
        else:
            self.depth = 0
        self.children = []

    def is_leaf(self):
        return len(self.children) == 0

    def create_state(self):
        self.state = self.parent.state.transition(self.action)

    def get_action_values(self):
        return np.array([child.total_rews/(child.num_visits+1e-12) for child in self.children])

    def select_child(self, c):
        uct_values = np.array([child.puct(child.num_visits, self.num_visits, child.total_rews, c, prior) for (child, prior) in zip(self.children, self.priors)])
        return self.children[self.select_child_jit(uct_values)]

    @staticmethod
    @jit(nopython=True, cache=True)
    def select_child_jit(uct_values):
        max_uct_indices = np.flatnonzero(uct_values == np.max(uct_values))

        if len(max_uct_indices) == 1:
            return max_uct_indices[0]
        return np.random.choice(max_uct_indices)

    @staticmethod
    @jit(nopython=True, cache=True)
    def puct(num_visits, parent_visits, total_rews, c, prior):
        if num_visits == 0:
            return np.inf
        return total_rews/(num_visits+1e-12) + c * prior * np.sqrt(parent_visits) / (1 + num_visits)

class DirichletNode(PUCTNode):
    """PUCT perturbed with Dirichlet noise for root node."""
    def __init__(self, state, action=None, parent=None):
        super().__init__(state, action, parent)
        self.eps = 0.0
        self.dir_noise = 1.0

    def select_child(self, c):
        perturbed_priors = (1-self.eps) * self.priors + self.eps * np.random.dirichlet([self.dir_noise] * self.priors.shape[0])
        uct_values = np.array([child.puct(child.num_visits, self.num_visits, child.total_rews, c, prior) for (child, prior) in zip(self.children, perturbed_priors)])
        return self.children[self.select_child_jit(uct_values)]

