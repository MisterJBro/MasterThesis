import time
import numpy as np
from copy import deepcopy
from numba import jit


class Tree:
    """ MCTS Tree presentation. Using index tree. """
    def __init__(self, state, num_players=2):
        self.num_players=num_players
        self.root = Node(state)

    def search(self, iters=1000):
        iter = 0

        while iter < iters:
            leaf = self.select()
            new_leaf = self.expand(leaf)
            ret = self.simulate(new_leaf)
            self.backpropagate(new_leaf, ret)
            iter += 1
        print(np.round(self.get_action_values(), 2))

    def select(self):
        node = self.root
        while not node.is_leaf():
            node = node.select_child()
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
            new_node = Node(None, action=action, parent=node)
            node.children.append(new_node)

        # Pick child node
        child = np.random.choice(node.children)
        child.create_state()
        return child

    def simulate(self, node):
        return node.rollout(self.num_players)

    def backpropagate(self, node, ret):
        node.backpropagate(ret, self.num_players)

    def get_action_values(self):
        return self.root.get_action_values()

class Node:
    """ Tree Node abstraction. All values are stored in the tree. """

    def __init__(self, state, action=None, parent=None):
        self.state = state
        self.num_visits = 0
        self.total_rews = 0.0

        self.action = action
        self.parent = parent
        if parent is not None:
            self.depth = parent.depth + 1
        else:
            self.depth = 0
        self.children = []

    def is_leaf(self):
        return len(self.children) == 0

    def select_child(self):
        uct_values = np.array([child.uct() for child in self.children])
        return self.children[self.select_child_jit(uct_values)]

    @staticmethod
    @jit(nopython=True, cache=True)
    def select_child_jit(uct_values):
        max_uct_indices = np.flatnonzero(uct_values == np.max(uct_values))

        if len(max_uct_indices) == 1:
            return max_uct_indices[0]
        return np.random.choice(max_uct_indices)

    def uct(self):
        if self.num_visits == 0:
            return float('inf')
        return self.uct_jit(self.num_visits, self.parent.num_visits, self.total_rews)

    @staticmethod
    @jit(nopython=True, cache=True)
    def uct_jit(num_visits, parent_visits, total_rews):
        return total_rews/(num_visits+1e-12) + np.sqrt(2) * np.sqrt(np.log(parent_visits)/num_visits+1e-12)

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

class State:
    """ State representation of the environment. """
    def __init__(self, env, done=False, rew=0.0):
        self.env = env
        self.done = done
        self.rew = rew

    def transition_inplace(self, action):
        _, self.rew, self.done, _ = self.env.step(action)
        self.rew = np.abs(self.rew)

    def transition(self, action):
        next_state = deepcopy(self)
        next_state.transition_inplace(action)
        return next_state

    def get_possible_actions(self):
        return self.env.available_actions()

    def is_terminal(self):
        return self.done

    def rollout(self, num_players):
        env = deepcopy(self.env)
        done = self.done
        ret = self.rew
        player = 0
        while not done:
            player = (player + 1) % num_players
            act = np.random.choice(env.available_actions())
            _, rew, done, _ = env.step(act)

            if player == 0:
                ret += np.abs(rew)
            else:
                ret -= np.abs(rew)
        return ret

    def __str__(self):
        return str(self.env)

if __name__ == "__main__":
    from gym_tictactoe.env import TicTacToeEnv

    env = TicTacToeEnv(show_number=True)
    state = env.reset()

    env.step(0)
    env.step(4)
    env.step(1)

    done = False
    while not done:
        tree = Tree(State(env), num_players=2)
        tree.search(iters=2_000)

        act = np.random.choice(env.available_actions())
        state, reward, done, info = env.step(act)

        env.render()
        quit()