from copy import deepcopy
import time
import numpy as np
from multiprocessing import freeze_support
from src.networks.policy_pend import PendulumPolicy
from src.train.config import create_config
from src.search.alpha_zero import AlphaZero
from src.env.discretize_env import DiscreteActionWrapper
from src.env.pendulum import PendulumEnv
from src.search.state import State
from src.train.exit import AZExitTrainer
from src.train.pg import PGTrainer


if __name__ == '__main__':
    env = DiscreteActionWrapper(PendulumEnv(), n_bins=11)
    config = create_config({
        "env": env,
        "puct_c": 3.0,
        "train_iters": 150,
        "az_iters": 200,
        "az_eval_batch": 15,
        "num_cpus": 3,
        "num_envs": 30,
        "num_trees": 15,
        "device": "cpu",
        "pi_lr": 8e-4,
        "vf_lr": 5e-4,
        "vf_iters": 5,
        "sample_len": 500,
    })

    with AZExitTrainer(config) as tr:
        tr.train()
        tr.test()
        tr.save()
    quit()

    freeze_support()
    policy = PendulumPolicy(config)
    az = AlphaZero(policy, config)

    obs = env.reset()
    env.env.state = np.array([np.pi, 0.0])

    start = time.time()
    done = False
    iter = 0
    ret = 0
    while not done:
        print(iter)
        az.update_policy(policy.state_dict())
        qvals = az.search(State(env, obs=obs))

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
