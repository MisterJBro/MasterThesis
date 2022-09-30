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

    def reset(self):
        self.is_black = True
        return self.env.reset()

    def step(self, action):
        self.is_black = not self.is_black
        action = np.argwhere(action)[0][0]
        obs, rew, done = self.env.step(action)

        if not self.is_black:
            obs[[0, 1]] = obs[[1, 0]]
        return obs, rew, done

    def available_actions(self):
        return np.eye(self.size*self.size)

    def __str__(self):
        return str(self.env)
