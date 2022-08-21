import gym
import numpy as np


class DiscreteActionWrapper(gym.ActionWrapper):
    """Discretizes the action space of an gym continuous environment."""

    def __init__(self, env, n_bins=11):
        super().__init__(env, new_step_api=True)
        self.n_bins = n_bins
        self.converter = np.linspace(-2, 2, num=n_bins)
        self.action_space = gym.spaces.Discrete(n_bins)

    def action(self, action):
        """ Discrete to continuous action."""
        action = self.converter[action]
        return np.array([action])

    def reverse_action(self, action):
        """ Continuous to discrete action."""
        action = np.argwhere(self.converter == action)[0][0]
        return action

    def available_actions(self):
        return np.arange(self.n_bins)
