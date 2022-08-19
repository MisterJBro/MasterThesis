from copy import deepcopy
import gym
from pendulum import PendulumEnv
import numpy as np
from multiprocessing import freeze_support
from mcts import MCTS
from state import State


# Discretizes the action space of an gym environment
class DiscreteActionWrapper(gym.ActionWrapper):
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

if __name__ == "__main__":
    freeze_support()

    config = {
        "uct_c": np.sqrt(2),
        "mcts_iters": 1_000,
        "num_trees": 4,
        "bandit_policy": "uct",
        "num_players": 1,
    }

    env = DiscreteActionWrapper(PendulumEnv())
    obs = env.reset()
    env.env.state = np.array([np.pi, 0.0])

    mcts = MCTS(config)
    print("Test search")
    qvals = mcts.search(State(env), iters=10)
    print("Finished")

    done = False
    iter = 0
    ret = 0
    while not done:
        import time
        start = time.time()
        qvals = mcts.search(State(env))

        act = env.available_actions()[np.argmax(qvals)]
        obs, reward, done, info = env.step(act)
        print(f"Reward: {reward:0.2f}  Time: {time.time() - start:0.2f}s")
        ret += reward

        render_env = deepcopy(env)
        render_env.render()
        iter += 1
    print(ret)
    render_env.close()
    env.close()
    mcts.close()
