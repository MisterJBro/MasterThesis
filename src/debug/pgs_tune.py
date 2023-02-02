import os
import torch
from torch.multiprocessing import freeze_support
from src.train.config import create_config
from src.env.hex import HexEnv
from src.networks.residual import HexPolicy
from src.search.mcts.mcts import MCTS
import numpy as np
from src.search.pgs.pgs import PGS
import math
from src.search.state import State
from hyperopt import hp, tpe, fmin, Trials


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    freeze_support()

    # Space
    space = {
        'puct_c': hp.choice('puct_c', [5.0]),
        'trunc_len': hp.choice('trunc_len', [4, 5, 6]),
        'pgs_lr': hp.choice('pgs_lr', [1e-1, 4e-2, 1e-2, 6e-3, 2e-3, 6e-4]),
        'entr_c': hp.choice('entr_c', [0.1, 0.01, 0.2, 0.5]),
        'kl_c': hp.choice('kl_c', [0.1, 0.01, 0.2, 0.5]),
        'p_val': hp.choice('p_val', [0.2, 0.8, 0.95, 0.995]),
    }

    # Init for algos
    size = 5
    env = HexEnv(size)
    config = create_config({
        "env": env,
        "device": "cpu",
        "search_num_workers": 1,
        "search_evaluator_batch_size": 1,
        "device": "cpu",
    })
    config["num_res_blocks"] = 8
    config["num_filters"] = 128

    # Paths
    path1 = "../../checkpoints/p_5x5_8_128.pt"

    # Policies
    policy1 = HexPolicy(config)
    policy1.load(path1)
    policy1.eval()

    def get_action(env, obs, info, iters, p_searcher):
        result = p_searcher.search(State(env, obs=obs), iters=iters)
        # Sample from pi
        pi = result["pi"].reshape(-1)
        #print("A", pi)
        #pi[pi == 0] = -10e8
        #print(pi)
        #pi = (np.exp(pi) / np.sum(np.exp(pi))).reshape(-1)
        #print(pi)
        print(result["visits"])
        act = np.random.choice(len(pi), p=pi)
        return act

        #act = np.argmax(result["pi"])
        #return act

    # Simulate
    def simulate(params):
        # PGS
        pgs = PGS(config, policy1, puct_c=params["puct_c"], trunc_len=params["trunc_len"], pgs_lr=params["pgs_lr"], entr_c=params["entr_c"], kl_c=params["kl_c"], p_val=params["p_val"])
        mcs = PGS(config, policy1, mcs=True, puct_c=params["puct_c"], trunc_len=params["trunc_len"], pgs_lr=params["pgs_lr"], entr_c=params["entr_c"], kl_c=params["kl_c"], p_val=params["p_val"])
        iters = 200
        num_victories = 0
        num_games = 100
        for pid in range(2):
            for _ in range(num_games):
                obs, info = env.reset()
                env.render()

                done = False
                black_turn = True
                while not done:
                    # Action
                    if black_turn:
                        if pid == 0:
                            act = get_action(env, obs, info, iters, pgs)
                        else:
                            act = get_action(env, obs, info, iters, mcs)
                    else:
                        if pid == 0:
                            act = get_action(env, obs, info, iters, mcs)
                        else:
                            act = get_action(env, obs, info, iters, pgs)
                    obs, rew, done, info = env.step(act)

                    black_turn = not black_turn

                black_turn = not black_turn
                if (black_turn and pid == 0) or (not black_turn and pid == 1):
                    num_victories += rew
        win_rate = num_victories / (num_games*2)
        return win_rate

    # Objective
    def objective(params):
        win_rate = simulate(params)

        # Write progress
        f = open("tune_log.txt", "a")
        f.write(f'Winrate: {win_rate}:\nParams: {params}\n')
        f.close()

        return -win_rate

    # Trials
    trials = Trials()

    # Optimizing
    best = fmin(objective,
                space = space, algo=tpe.suggest,
                max_evals = 200, trials=trials)

    # Print results
    print(best)
    print(trials.results)