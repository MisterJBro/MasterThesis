from copy import deepcopy
import time
import numpy as np
from torch.multiprocessing import freeze_support
from src.networks.residual import PendulumPolicy
from src.train.config import create_config
from src.search.alpha_zero.alpha_zero import AlphaZero
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
        "az_iters": 200,
        "az_eval_batch": 1,
        "dirichlet_eps": 0.0,
        "search_return_adv": False,
    })

    freeze_support()
    policy = PendulumPolicy(config)
    policy.load("checkpoints/policy_pdlm_pgtrainer.pt")
    az = AlphaZero(config, policy)
    obs = env.reset()
    import time
    start = time.time()

    done = False
    iter = 0
    ret = 0
    while not done:
        #az.update_policy(policy.state_dict())
        qvals = az.search(State(env, obs=obs))

        act = env.legal_actions()[np.argmax(qvals)]
        obs, reward, done, info = env.step(act)
        ret += reward

        #deepcopy(env).render()
        iter += 1
    print(ret)
    print("Time: ", time.time()-start)
    env.close()
    az.close()