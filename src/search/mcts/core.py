import numpy as np
from src.search.node import UCTNode


class MCTSCore:
    """ Search Tree for MCTS. """

    def __init__(self, state, config):
        self.config = config
        self.num_players = config["num_players"]
        self.NodeClass = UCTNode
        self.expl_coeff = config["uct_c"]
        self.set_root(state)

    def search(self, iters, default=-1e9):
        iter = 0

        while iter < iters:
            leaf = self.select()
            new_leaf = self.expand(leaf)
            ret = self.simulate(new_leaf)
            self.backpropagate(new_leaf, ret)
            iter += 1

        qvals = self.root.get_action_values(self.config["num_acts"], default=default)
        val = -self.root.qvalue()
        return {
            "pi": qvals,
            "Q": qvals,
            "V": val,
        }

    def select(self):
        node = self.root
        while not node.is_leaf():
            node = node.select_child(self.expl_coeff)
        return node

    def expand(self, node):
        if node.state is None:
            node.create_state()
            return node
        if node.is_terminal():
            return node

        # Create new child nodes, lazy init
        actions = node.get_legal_actions()
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
