import gym
import numpy as np

# Discretizes the action space of an gym environment
class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env, n_bins=5):
        super().__init__(env)
        self.n_bins = n_bins
        self.converter = np.linspace(-1, 1, num=n_bins)
        self.action_space = gym.spaces.MultiDiscrete([n_bins]*4)

    def action(self, action):
        """ Discrete to continous action."""
        action = self.converter[action]
        #action = action.reshape(4, -1)
        #action *= self.converter
        #action = action.sum(-1)
        return action

    def reverse_action(self, action):
        """ Continouos action to discrete."""
        action = np.array([np.argwhere(self.converter == a)[0][0] for a in action])
        #action = np.eye(self.n_bins)[action]
        return action


env = DiscreteActionWrapper(gym.make("BipedalWalker-v3"))
obs = env.reset()

for _ in range(5000):
    env.render()
    act = env.action_space.sample()

    print(act)
    obs, rew, done, _ = env.step(act)

    if done:
        obs = env.reset()

env.close()