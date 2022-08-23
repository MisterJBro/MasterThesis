import numpy as np
from multiprocessing import Process
from src.search.node import UCTNode, PUCTNode, DirichletNode
from src.search.util import measure_time


class Tree:
    """ Search Tree for MCTS. """
    def __init__(self, state, config):
        self.config = config
        self.num_players = config["num_players"]
        self.NodeClass = UCTNode
        self.expl_coeff = config["uct_c"]
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

        return self.root.get_action_values()

    def select(self):
        node = self.root
        while not node.is_leaf():
            node = node.select_child(self.expl_coeff)
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
        return node.state.rollout(self.num_players, self.config["gamma"])

    def backpropagate(self, node, ret):
        curr_ret = ret
        gamma = self.config["gamma"]

        while node is not None:
            node.num_visits += 1
            curr_ret = node.state.rew + gamma * curr_ret
            node.total_rews += curr_ret

            flip = -1.0 if self.num_players == 2 else 1.0
            curr_ret *= flip
            node = node.parent

    def set_root(self, state):
        self.root = self.NodeClass(state)

class AZTree(Tree):
    """ Search Tree presentation for Alpha Zero. """

    def __init__(self, state, eval_channel, config, idx=0):
        self.config = config
        self.eval_channel = eval_channel
        self.idx = idx
        self.num_players = config["num_players"]
        self.NodeClass = PUCTNode
        self.expl_coeff = config["puct_c"]
        self.set_root(state)

    def search(self, iters=1_000):
        return super().search(iters)
        #return self.get_policy_targets()

    def simulate(self, node):
        probs, val = self.eval_fn(node.state.obs)
        node.priors = probs
        return val

    def set_root(self, state):
        self.root = DirichletNode(state, eps=self.config["dirichlet_eps"], noise=self.config["dirichlet_noise"])
        if state is not None:
            probs, _ = self.eval_fn(self.root.state.obs)
            self.root.priors = probs

    def eval_fn(self, obs):
        self.eval_channel.send({
            "obs": obs,
            "ind": self.idx,
        })
        msg = self.eval_channel.recv()
        return msg["probs"], msg["val"]

    def get_policy_targets(self, temp=1.0):
        return [child.num_visits ** (1/temp) / self.root.num_visits ** (1/temp) for child in self.root.children]


class TreeWorker(Process, Tree):
    """ Multiprocessing Tree Worker, for parallelization of MCTS."""
    def __init__(self, iters, config, channel):
        Tree.__init__(self, None, config)
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


class AZTreeWorker(Process, AZTree):
    """ Multiprocessing Tree Worker, for parallelization of MCTS."""
    def __init__(self, iters, eval_channel, idx, config, channel):
        AZTree.__init__(self, None, eval_channel, config, idx=idx)
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