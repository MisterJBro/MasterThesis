from copy import deepcopy
from pendulum import PendulumEnv
import numpy as np
from multiprocessing import freeze_support
from discretize_env import DiscreteActionWrapper
from policy import ActorCriticPolicy
from state import State
from alpha_zero import AZTree

if __name__ == "__main__":
    # Init
    freeze_support()
    config = {
        "uct_c": np.sqrt(2),
        "puct_c": 10.0,
        "mcts_iters": 1_000,
        "num_trees": 4,
        "bandit_policy": "puct",
        "num_players": 1,
        "pi_lr": 1e-3,
        "vf_lr": 5e-4,
        "flat_obs_dim": 3,
        "num_acts": 11,
        "device": "cpu",
    }

    env = DiscreteActionWrapper(PendulumEnv(), n_bins=config["num_acts"])
    policy = ActorCriticPolicy(config)
    obs = env.reset()

    env.env.state = np.array([np.pi, 0.0])

    done = False
    iter = 0
    ret = 0
    while not done:
        tree = AZTree(State(env, obs=obs), policy, config)
        qvals = tree.search(iters=1000)
        print(qvals)

        act = env.available_actions()[np.argmax(qvals)]
        obs, reward, done, info = env.step(act)
        ret += reward

        render_env = deepcopy(env)
        render_env.render()
        iter += 1
    print(ret)
    render_env.close()
    env.close()
