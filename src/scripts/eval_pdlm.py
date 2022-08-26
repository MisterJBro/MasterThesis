from copy import deepcopy
import time
import torch
import numpy as np
from multiprocessing import freeze_support
from src.networks.policy_pend import PendulumPolicy
from src.search.pgs import PGS
from src.train.config import create_config
from src.search.alpha_zero import AlphaZero
from src.search.mcts import MCTS
from src.env.discretize_env import DiscreteActionWrapper
from src.env.pendulum import PendulumEnv
from src.search.state import State
from tqdm import tqdm


def eval(env, get_action, iters, render=False):
    obs = env.reset()

    done = False
    ret = 0
    while not done:
        act = get_action(env, obs, iters)
        obs, reward, done, _ = env.step(act)
        ret += reward

        if render:
            render_env = deepcopy(env)
            render_env.render()
    if render:
        render_env.close()
    return ret


if __name__ == '__main__':
    env = DiscreteActionWrapper(PendulumEnv())
    config = create_config({
        "env": env,
        "puct_c": 20.0,
        "uct_c": 5.0,
        "mcts_iters": 500,
        "az_iters": 500,
        "pgs_iters": 500,
        "az_eval_batch": 1,
        "dirichlet_eps": 0.0,
        "pgs_lr": 2e-5,
        "pgs_trunc_len": 5,
        "num_trees": 1,
        "device": "cpu",
        "tree_output_qvals": True,
    })

    freeze_support()
    policy = PendulumPolicy(config)
    policy.load("checkpoints/policy_pdlm_pgtrainer.pt")
    mcts_obj = MCTS(config)
    az_obj = AlphaZero(policy, config)
    pgs_obj = PGS(policy, config)

    def nn(env, obs, iters):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            dist = policy.get_dist(obs)
        act = dist.logits.argmax(-1)

        return act.cpu().numpy()

    def mcts(env, obs, iters):
        qvals = mcts_obj.search(State(env, obs=obs), iters=iters)

        act = env.available_actions()[np.argmax(qvals)]
        return act

    def az(env, obs, iters):
        az_obj.update_policy(policy.state_dict())
        qvals = az_obj.search(State(env, obs=obs), iters=iters)

        act = env.available_actions()[np.argmax(qvals)]
        return act

    def pgs(env, obs, iters):
        pgs_obj.update_policy(policy.state_dict())
        qvals = pgs_obj.search(State(env, obs=obs), iters=iters)

        act = env.available_actions()[np.argmax(qvals)]
        return act

    # Eval
    algos = [pgs]
    for iters in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        for algo in algos:
            rets = []
            for _ in tqdm(range(1), ncols=100, desc=f'{iters}'):
                ret = eval(env, algo, iters, render=False)
                rets.append(ret)
            print(f'Iters: {iters} - Algo: {algo.__name__.upper()} - Return: {np.mean(rets):.03f} - Std dev: {np.std(rets):.03f}')

    # Close
    env.close()
    mcts_obj.close()
    az_obj.close()
    pgs_obj.close()