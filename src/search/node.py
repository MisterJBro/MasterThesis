from abc import ABC, abstractmethod
from tkinter import N
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

    def get_legal_actions(self):
        return self.state.get_possible_actions()

    def get_action_values(self, default=0):
        return np.array([child.total_rews/(child.num_visits+1e-12) if child.num_visits > 0 else default for child in self.children])


class UCTNode(Node):
    """ Tree Node using Upper Confidence bound1 for Trees (UCT). """

    def __init__(self, state, action=None, parent=None):
        super().__init__(state, action, parent)

        self.num_visits = 0
        self.total_rews = 0.0

    def select_child(self, c):
        uct_values = np.array([child.uct(child.num_visits, self.num_visits, child.total_rews, c) for child in self.children])
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
    def uct(num_visits, parent_visits, total_rews, c):
        if num_visits == 0:
            return np.inf
        return total_rews/(num_visits+1e-12) + c * np.sqrt(np.log(parent_visits)/num_visits+1e-12)


class PUCTNode(Node):
    """Predictor + UCB for Trees"""

    def __init__(self, state, action=None, parent=None):
        super().__init__(state, action, parent)

        self.num_visits = 0
        self.total_rews = 0.0
        self.priors = None

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
    """PUCT node where prior is perturbed with Dirichlet noise for better exploration."""

    def __init__(self, state, action=None, parent=None, eps=0.25, noise=1.0):
        super().__init__(state, action, parent)
        self.eps = eps
        self.dir_noise = noise

    def select_child(self, c):
        perturbed_priors = (1-self.eps) * self.priors + self.eps * np.random.dirichlet([self.dir_noise] * self.priors.shape[0])
        uct_values = np.array([child.puct(child.num_visits, self.num_visits, child.total_rews, c, prior) for (child, prior) in zip(self.children, perturbed_priors)])
        return self.children[self.select_child_jit(uct_values)]


class PGSNode(PUCTNode):
    """Changes made for PGS."""

    def __init__(self, state, action=None, parent=None):
        super().__init__(state, action, parent)
        self.qval = 0

    def select_child(self, c):
        uct_values = np.array([child.puct(child.qvalue(), child.num_visits, self.num_visits, c, prior) for (child, prior) in zip(self.children, self.priors)])
        return self.children[self.select_child_jit(uct_values, self.priors)]

    @staticmethod
    @jit(nopython=True, cache=True)
    def select_child_jit(uct_values, priors):
        max_uct_indices = np.flatnonzero(uct_values == np.max(uct_values))

        if len(max_uct_indices) == 1:
            return max_uct_indices[0]
        max_prior_indices = np.flatnonzero(priors[max_uct_indices] == np.max(priors[max_uct_indices]))
        if len(max_prior_indices) == 1:
            return max_uct_indices[max_prior_indices[0]]
        return max_uct_indices[np.random.choice(max_prior_indices)]

    @staticmethod
    @jit(nopython=True, cache=True)
    def puct(qval, num_visits, parent_visits, c, prior):
        if num_visits == 0:
            return np.inf
        return qval + c * prior * np.sqrt(parent_visits) / (1 + num_visits)

    def qvalue(self):
        return self.total_rews/(self.num_visits+1e-12) # self.qval #

    def get_action_values(self, default=0):
        return np.array([child.qvalue() if child.num_visits > 0 else default for child in self.children])
