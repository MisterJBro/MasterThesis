import gym
import numpy as np
from hexgame import RustEnv


class HexEnv:
    """Hex wrapper."""

    def __init__(self, size=9):
        self.size = size
        self.is_black = True
        self.env = RustEnv(size)
        self.action_space = gym.spaces.Discrete(size*size)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(2, size, size), dtype=np.float32)
        self.num_players = 2
        self.converter = np.eye(size*size)

    def reset(self):
        self.is_black = True
        return self.env.reset()

    def step(self, action):
        self.is_black = not self.is_black
        obs, rew, done, legal_act = self.env.step(action)

        if not self.is_black:
            obs[[0, 1]] = obs[[1, 0]]
            obs[0] = obs[0].T
            obs[1] = obs[1].T
        return obs, rew, done, {"pid": int(not self.is_black), "legal_act": legal_act}

    def render(self):
        print(self.env)

    def legal_actions(self):
        return self.env.legal_actions()

    def seed(self, value):
        pass

    def close(self):
        pass

    def __str__(self):
        return "hex"

    def __getstate__(self):
        return (self.size, self.is_black, self.action_space, self.observation_space, self.env.to_pickle())

    def __setstate__(self, state):
        self.size, self.is_black, self.action_space, self.observation_space, pickle = state
        self.env = RustEnv(self.size)
        self.env.from_pickle(pickle)
