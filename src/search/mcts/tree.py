import numpy as np
from src.search.node import UCTNode
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

    def get_normalized_visit_counts(self, temp=1.0):
        return [child.num_visits ** (1/temp) / self.root.num_visits ** (1/temp) for child in self.root.children]
