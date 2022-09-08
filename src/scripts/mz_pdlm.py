from copy import deepcopy
import time
import numpy as np
from multiprocessing import freeze_support
from src.networks.model import ValueEquivalenceModel
from src.networks.policy_pend import PendulumPolicy
from src.train.config import create_config
from src.search.mu_zero.mu_zero import MuZero
from src.env.discretize_env import DiscreteActionWrapper
from src.env.pendulum import PendulumEnv
from src.search.state import ModelState


if __name__ == '__main__':
    freeze_support()
    env = DiscreteActionWrapper(PendulumEnv())
    config = create_config({
        "env": env,
        "num_trees": 1,
        "device": "cpu",
        "puct_c": 5.0,
        "mz_iters": 200,
        "mz_eval_batch": 1,
        "dirichlet_eps": 0.0,
        "tree_output_qvals": False,
    })

    policy = PendulumPolicy(config)
    policy.load("checkpoints/policy_pdlm_pgtrainer.pt")
    model = ValueEquivalenceModel(config)
    mz = MuZero(model, policy, config)
    obs = env.reset()
    import time
    start = time.time()

    done = False
    iter = 0
    ret = 0
    while not done:
        qvals = mz.search(ModelState(env, obs=obs))

        act = env.available_actions()[np.argmax(qvals)]
        obs, reward, done, info = env.step(act)
        ret += reward

        deepcopy(env).render()
        iter += 1
    print(ret)
    print("Time: ", time.time()-start)
    env.close()
    mz.close()