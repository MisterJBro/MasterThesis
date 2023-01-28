import gym
import numpy as np
from hexgame import RustEnv


class HexEnv:
    """Hex wrapper."""

    def __init__(self, size=9):
        self.size = size
        self.env = RustEnv(size)
        self.action_space = gym.spaces.Discrete(size*size)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(2, size, size), dtype=np.float32)
        self.num_players = 2
        self.converter = np.eye(size*size)

    def reset(self):
        obs, info = self.env.reset()
        info["legal_act"] = np.array(info["legal_act"])

        return obs, info

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        info["legal_act"] = np.array(info["legal_act"])

        return obs, rew, done, info

    def render(self):
        print(self.env)

    def legal_actions(self):
        return self.env.legal_actions()

    def seed(self, value):
        pass

    def close(self):
        pass

    def __str__(self):
        return f"hex(size={self.size})"

    def __getstate__(self):
        return (self.size, self.action_space, self.observation_space, self.env.to_pickle())

    def __setstate__(self, state):
        self.size, self.action_space, self.observation_space, pickle = state
        self.env = RustEnv(self.size)
        self.env.from_pickle(pickle)
