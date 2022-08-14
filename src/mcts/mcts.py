import time
import numpy as np
from copy import deepcopy
from numba import jit, float32, int32, boolean


class Tree:
    """ MCTS Tree presentation. Using index tree. """
    def __init__(self, state, num_players=2):
        self.num_players=num_players
        self.root = Node(state)

    def search(self, iters=1000):
        iter = 0
        time_select = 0.0
        time_expand = 0.0
        time_simulate = 0.0
        time_backpropagate = 0.0

        while iter < iters:
            start = time.time()
            leaf = self.select()
            time_select += time.time() - start
            start = time.time()
            new_leaf = self.expand(leaf)
            time_expand += time.time() - start

            if new_leaf.state.is_terminal():
                iter += 1
                continue

            start = time.time()
            ret = self.simulate(new_leaf)
            time_simulate += time.time() - start
            start = time.time()
            self.backpropagate(new_leaf, ret)
            time_backpropagate += time.time() - start
            iter += 1
        print(np.round(self.get_action_values(), 2).reshape(3, 3))
        print(f"Times: {time_select:.2f}s {time_expand:.2f}s {time_simulate:.2f}s {time_backpropagate:.2f}s")

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
        child = node.children[np.random.randint(len(node.children))]
        child.create_state()
        return child

    def simulate(self, node):
        return node.rollout(self.num_players)

    def backpropagate(self, node, ret):
        node.backpropagate(ret, self.num_players)

    def get_action_values(self):
        return self.root.get_action_values(self.num_players)

class Node:
    """ Tree Node abstraction. All values are stored in the tree. """

    def __init__(self, state, action=None, parent=None):
        self.state = state
        self.num_visits = 1e-12
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
        max_uct_values = max(uct_values)
        max_uct_indices = [i for i, x in enumerate(uct_values) if x == max_uct_values]

        if len(max_uct_indices) == 1:
            return max_uct_indices[0]
        return np.random.choice(np.array(max_uct_indices))

    def uct(self):
        if self.num_visits == 0:
            return float('inf')
        elif self.state is not None and self.state.is_terminal():
            return -float('inf')
        return self.uct_jit(self.num_visits, self.parent.num_visits, self.total_rews)

    @staticmethod
    @jit(nopython=True, cache=True)
    def uct_jit(num_visits, parent_visits, total_rews):
        return total_rews/num_visits + 1.4142 * np.sqrt(np.log(parent_visits)/num_visits)

    def create_state(self):
        self.state = self.parent.state.copy()
        self.state.make_action(self.action)

    def rollout(self, num_players):
        return self.state.rollout(self.depth, num_players)

    def backpropagate(self, ret, num_players):
        self.num_visits += 1
        self.total_rews += ret

        if self.parent is not None:
            player = self.depth % num_players
            flip = -1.0 if (player == 0 and num_players > 1) or player == 1 else 1.0
            self.parent.backpropagate(flip * ret, num_players)

    def get_action_values(self, num_players):
        player = (self.depth+1) % num_players
        flip = 1.0 if player == 0 else -1.0
        return flip * np.array([child.total_rews/child.num_visits for child in self.children])

class State:
    """ State representation of the environment. """
    def __init__(self, env, done=False):
        self.env = env
        self.done = done

    def copy(self):
        return deepcopy(self)

    def make_action(self, action):
        _, _, self.done, _ = self.env.step(action)

    def get_possible_actions(self):
        return self.env.available_actions()

    def is_terminal(self):
        return self.done

    def rollout(self, depth, num_players):
        env = deepcopy(self.env)
        done = self.done
        ret = 0
        player = 0
        while not done:
            act = np.random.choice(env.available_actions())
            obs, rew, done, _ = env.step(act)

            if player == 0:
                ret += np.abs(rew)
            else:
                ret -= np.abs(rew)
            player = (player + 1) % num_players
        return ret

    def __str__(self):
        return str(self.env)

if __name__ == "__main__":
    from gym_tictactoe.env import TicTacToeEnv

    env = TicTacToeEnv(show_number=True)
    state = env.reset()

    tree = Tree(State(env), num_players=2)
    tree.search(iters=2)

    done = False
    while not done:
        tree = Tree(State(env), num_players=2)
        tree.search(iters=10_000)

        act = np.random.choice(env.available_actions())

        state, reward, done, info = env.step(act)
        env.render()
        quit()