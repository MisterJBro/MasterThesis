from copy import deepcopy
import time
import numpy as np
from multiprocessing import freeze_support
from src.networks.policy_pend import PendulumPolicy
from src.train.config import create_config
from src.search.pgs import PGS
from src.env.discretize_env import DiscreteActionWrapper
from src.env.pendulum import PendulumEnv
from src.search.state import State


if __name__ == '__main__':
    env = DiscreteActionWrapper(PendulumEnv())
    config = create_config({
        "env": env,
        "num_trees": 1,
        "device": "cpu",
        "puct_c": 5.0,
        "pgs_lr": 1e-4,
        "pgs_trunc_len": 10,
        "pgs_iters": 200,
        "pgs_eval_batch": 1,
        "tree_output_qvals": False,
    })

    freeze_support()
    policy = PendulumPolicy(config)
    policy.load("checkpoints/policy_pdlm_pgtrainer.pt")
    pgs = PGS(policy, config)
    obs = env.reset()

    done = False
    iter = 0
    ret = 0
    while not done:
        pgs.update_policy(policy.state_dict())
        qvals = pgs.search(State(env, obs=obs))

        act = env.available_actions()[np.argmax(qvals)]
        obs, reward, done, info = env.step(act)
        ret += reward

        deepcopy(env).render()
        iter += 1
    print(ret)
    env.close()
    pgs.close()