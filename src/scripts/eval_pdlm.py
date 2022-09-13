import argparse
from copy import deepcopy
import os
import time
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import freeze_support
from src.networks.policy_pend import PendulumPolicy
from src.networks.model import ValueEquivalenceModel
from src.search.pgs.pgs import PGS
from src.train.config import create_config
from src.search.alpha_zero.alpha_zero import AlphaZero
from src.search.mu_zero.mu_zero import MuZero
from src.search.mcts.mcts import MCTS
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

parser = argparse.ArgumentParser(description='File to evaluate methods')

# Parser arguments
parser.add_argument('--job_id', type=int, default=0)

if __name__ == '__main__':
    freeze_support()
    args = parser.parse_args()
    job_id = args.job_id

    # Init for algos
    env = DiscreteActionWrapper(PendulumEnv())
    config = create_config({
        "env": env,
        "puct_c": 20.0,
        "uct_c": 20.0,
        "mcts_iters": 500,
        "az_iters": 500,
        "az_eval_batch": 1,
        "dirichlet_eps": 0.0,
        "mz_iters": 500,
        "mz_eval_batch": 1,
        "pgs_lr": 1e-1,
        "pgs_iters": 1,
        "pgs_trunc_len": 5,
        "num_trees": 1,
        "device": "cpu",
        "tree_output_qvals": True,
    })

    # Import policy and model
    policy = PendulumPolicy(config)
    policy.load("checkpoints/policy_pdlm_pgtrainer.pt")
    model = ValueEquivalenceModel(config)
    model.load("checkpoints/ve_model.pt")

    # Algorithms
    mcts_obj = MCTS(config)
    az_obj = AlphaZero(policy, config)
    mz_obj = MuZero(model, policy, config)
    pgs_obj = PGS(policy, config)
    mcs_config = deepcopy(config)
    mcs_config.update({"pgs_lr": 0})
    mcs_obj = PGS(policy, mcs_config)

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
        qvals = az_obj.search(State(env, obs=obs), iters=iters)
        act = env.available_actions()[np.argmax(qvals)]
        return act

    def mz(env, obs, iters):
        qvals = mz_obj.search(State(env, obs=obs), iters=iters)
        act = env.available_actions()[np.argmax(qvals)]
        return act

    def mcs(env, obs, iters):
        qvals = mcs_obj.search(State(env, obs=obs), iters=iters)
        act = env.available_actions()[np.argmax(qvals)]
        return act

    def pgs(env, obs, iters):
        qvals = pgs_obj.search(State(env, obs=obs), iters=iters)
        act = env.available_actions()[np.argmax(qvals)]
        return act

    # Eval
    algos = [mcts, nn, az, mz, mcs, pgs]
    ret_iters = []
    all_iters = [10, 20, 40, 60, 80, 100, 200]#, , 400, 600, 800, 1000]
    curr_iters = all_iters[job_id]
    for iters in all_iters:#[curr_iters]:
        for algo in algos:
            rets = []
            for _ in tqdm(range(3), ncols=100, desc=f'{iters}'):
                ret = eval(env, algo, iters, render=False)
                rets.append(ret)
            print(f'Iters: {iters} - Algo: {algo.__name__.upper()} - Return: {np.mean(rets):.03f} - Std dev: {np.std(rets):.03f}')
            ret_iters.append(np.mean(rets))
    ret_iters = np.array(ret_iters).reshape(-1, len(algos)).T
    #print(f'Print all returns: {np.round(ret_iters, 2).tolist()}')

    # Plot
    sns.set_theme()
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    x = np.arange(len(all_iters))
    width = 0.6
    pos = np.linspace(0.0, width, num=len(algos))
    algo_names = [a.__name__.upper() for a in algos]

    for i in range(len(algos)):
        plt.bar(x - width/2 + pos[i], ret_iters[i], width/len(algos), label=algo_names[i])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.xlabel('Iters per Action')
    plt.ylabel('Return')
    plt.title('Pendulum Task - Algorithm comparison')
    plt.xticks(x, all_iters)
    plt.legend()
    plt.show()

    # Close
    env.close()
    mcts_obj.close()
    az_obj.close()
    mz_obj.close()
    pgs_obj.close()
    mcs_obj.close()
