from abc import ABC, abstractmethod
import numpy as np
from numba import jit


class Node(ABC):
    """ Abstract node class."""
    def __init__(self, state, action=None, parent=None):
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

    @abstractmethod
    def select_child(self, c):
        pass

    def create_state(self):
        self.state = self.parent.state.transition(self.action)

    def rollout(self, num_players):
        return self.state.rollout(num_players)

    def backpropagate(self, ret, num_players):
        self.num_visits += 1
        self.total_rews += ret

        if self.parent is not None:
            flip = -1.0 if num_players == 2 else 1.0

            self.parent.backpropagate(flip * ret, num_players)

    def get_action_values(self):
        return np.array([child.total_rews/(child.num_visits+1e-12) for child in self.children])


class UCTNode(Node):
    """ Tree Node using Upper Confidence bound1 for Trees (UCT). """

    def __init__(self, state, action=None, parent=None):
        super().__init__(state, action, parent)

        self.num_visits = 0
        self.total_rews = 0.0

    def select_child(self, c):
        uct_values = np.array([child.uct(c) for child in self.children])
        return self.children[self.select_child_jit(uct_values)]

    @staticmethod
    @jit(nopython=True, cache=True)
    def select_child_jit(uct_values):
        max_uct_indices = np.flatnonzero(uct_values == np.max(uct_values))

        if len(max_uct_indices) == 1:
            return max_uct_indices[0]
        return np.random.choice(max_uct_indices)

    def uct(self, uct_c):
        if self.num_visits == 0:
            return float('inf')
        return self.uct_jit(self.num_visits, self.parent.num_visits, self.total_rews, uct_c)

    @staticmethod
    @jit(nopython=True, cache=True)
    def uct_jit(num_visits, parent_visits, total_rews, uct_c):
        return total_rews/(num_visits+1e-12) + uct_c * np.sqrt(np.log(parent_visits)/num_visits+1e-12)
