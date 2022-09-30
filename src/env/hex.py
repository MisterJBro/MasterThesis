import gym
import numpy as np
from hexgame import HexGame


class HexEnv:
    """Hex wrapper."""

    def __init__(self, size=5):
        self.size = size
        self.is_black = True
        self.env = HexGame(size)
        self.action_space = gym.spaces.Discrete(size)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(2, size, size), dtype=np.float32)

    def reset(self):
        self.is_black = True
        return self.env.reset()

    def step(self, action):
        self.is_black = not self.is_black
        action = np.argwhere(action)[0][0]
        obs, rew, done = self.env.step(action)

        if not self.is_black:
            obs[[0, 1]] = obs[[1, 0]]
        return obs, rew, done, {}

    def render(self):
        print(self)

    def available_actions(self):
        actions = self.env.available_actions()
        actions = [np.eye(self.size*self.size)[a] for a in actions]
        return actions

    def __str__(self):
        return str(self.env)

    def __getstate__(self):
        return (self.size, self.is_black, self.action_space, self.observation_space, self.env.to_pickle())

    def __setstate__(self, state):
        self.size, self.is_black, self.action_space, self.observation_space, pickle = state
        self.env = HexGame(self.size)
        self.env.from_pickle(pickle)
