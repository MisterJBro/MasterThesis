from copy import deepcopy
from pendulum import PendulumEnv
import numpy as np
from multiprocessing import freeze_support
from discretize_env import DiscreteActionWrapper
from policy import ActorCriticPolicy
from alpha_zero import AlphaZero
from state import State
import time

if __name__ == "__main__":
    # Init
    freeze_support()
    config = {
        "uct_c": np.sqrt(2),
        "puct_c": 3.0,
        "mcts_iters": 1000,
        "az_iters": 1000,
        "az_eval_batch": 3,
        "az_eval_timeout": 0.001,

        "num_trees": 3,
        "bandit_policy": "puct",
        "num_players": 1,
        "pi_lr": 1e-3,
        "vf_lr": 5e-4,
        "flat_obs_dim": 3,
        "num_acts": 11,
        "device": "cpu",
    }

    policy = ActorCriticPolicy(config)
    az = AlphaZero(policy, config)
    env = DiscreteActionWrapper(PendulumEnv(), n_bins=config["num_acts"])
    obs = env.reset()

    env.env.state = np.array([np.pi, 0.0])

    start = time.time()
    done = False
    iter = 0
    ret = 0
    while not done:
        az.update_policy(policy.state_dict())
        qvals = az.search(State(env, obs=obs))
        #print(qvals)

        act = env.available_actions()[np.argmax(qvals)]
        obs, reward, done, info = env.step(act)
        ret += reward

        render_env = deepcopy(env)
        render_env.render()
        iter += 1
    print(ret)
    print(time.time() - start)
    render_env.close()
    env.close()
    az.close()
