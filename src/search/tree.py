from concurrent.futures import process
import numpy as np
from multiprocessing import Process
from node import UCTNode


class Tree:
    """ Search Tree presentation. """
    def __init__(self, state, config):
        self.config = config
        self.num_players = config["num_players"]
        self.NodeClass = UCTNode if config["bandit_policy"] == "uct" else UCTNode
        self.set_root(state)

    def search(self, iters=1_000):
        iter = 0

        while iter < iters:
            leaf = self.select()
            new_leaf = self.expand(leaf)
            ret = self.simulate(new_leaf)
            self.backpropagate(new_leaf, ret)
            iter += 1

        return self.root.get_action_values()

    def select(self):
        node = self.root
        while not node.is_leaf():
            node = node.select_child(self.config["uct_c"])
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
        return node.rollout(self.num_players)

    def backpropagate(self, node, ret):
        node.backpropagate(ret, self.num_players)

    def set_root(self, state):
        self.root = self.NodeClass(state)


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