# PGS ablation study

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
from src.debug.util import seed_all


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    freeze_support()

    # Init for algos
    size = 5
    env = HexEnv(size)
    config = create_config({
        "env": env,
        "device": "cpu",
        "puct_c": 3.0,
        "search_num_workers": 1,
        "search_evaluator_batch_size": 1,
        "dirichlet_eps": 0.0,
        "pgs_lr": 2e-2,
        "pgs_trunc_len": 5,
        "device": "cpu",
        "search_return_adv": True,
    })
    config["num_res_blocks"] = 8
    config["num_filters"] = 128

    # Seed
    seed_all(int(config["job_id"]))

    # Paths
    path1 = "checkpoints/p_5x5_8_128_weak.pt"

    # Policies
    policy1 = HexPolicy(config)
    policy1.load(path1)
    policy1.eval()

    # PGS
    pgs_original = PGS(config, policy1, puct_c=5.0, pgs_lr=0e-1)
    #pgs_mcs = PGS(config, policy1, mcs=True, puct_c=5.0)
    #pgs_dyn_length = PGS(config, policy1, dyn_length=True)
    #pgs_scale_vals = PGS(config, policy1, scale_vals=True)
    #pgs_expl_entr = PGS(config, policy1, expl_entr=True)
    #pgs_expl_kl = PGS(config, policy1, expl_kl=True)
    #pgs_visit_counts = PGS(config, policy1, visit_counts=True)
    #pgs_update = PGS(config, policy1, update=True)

    def get_action(env, obs, info, iters, p_searcher):
        result = p_searcher.search(State(env, obs=obs), iters=iters)
        # Sample from pi
        pi = result["pi"]
        pi = np.exp(pi) / np.sum(np.exp(pi))
        #act = np.random.choice(len(pi), p=pi)
        #return act

        act = np.argmax(result["pi"])
        return act

    # Simulate
    #variants = [pgs_mcs]#, pgs_dyn_length, pgs_scale_vals, pgs_expl_entr, pgs_expl_kl, pgs_visit_counts, pgs_update]
    names = ["pgs_mcs", "pgs_dyn_length", "pgs_scale_vals", "pgs_expl_entr", "pgs_expl_kl", "pgs_visit_counts", "pgs_update", "pgs_all"]
    for i, name in enumerate(names):
        if i == 0:
            pgs_variant = PGS(config, policy1, puct_c=5.0, pgs_lr=0e-1)
        elif i == 1:
            pgs_variant = PGS(config, policy1, dyn_length=True, puct_c=5.0, pgs_lr=1e-1)
        elif i == 2:
            pgs_variant = PGS(config, policy1, scale_vals=True, puct_c=5.0, pgs_lr=1e-1)
        elif i == 3:
            pgs_variant = PGS(config, policy1, expl_entr=True, puct_c=5.0, pgs_lr=1e-1)
        elif i == 4:
            pgs_variant = PGS(config, policy1, expl_kl=True, puct_c=5.0, pgs_lr=1e-1)
        elif i == 5:
            pgs_variant = PGS(config, policy1, visit_counts=True, puct_c=5.0, pgs_lr=1e-1)
        elif i == 6:
            pgs_variant = PGS(config, policy1, update=True, puct_c=5.0, pgs_lr=1e-1)
        elif i == 7:
            pgs_variant = PGS(config, policy1, puct_c=5.0, pgs_lr=1e-1, dyn_length=True, scale_vals=True, expl_entr=True, expl_kl=True, visit_counts=True, update=True)


        print(f"Testing variant: {name} against baseline")

        for iters in [25, 50, 100, 200]:
            print("Iters: ", iters)
            num_victories = 0
            num_games = 20
            for pid in range(2):
                for i in range(num_games):
                    obs, info = env.reset()

                    done = False
                    black_turn = True
                    while not done:
                        #env.render()
                        # Action
                        if black_turn:
                            if pid == 0:
                                act = get_action(env, obs, info, iters, pgs_variant)
                            else:
                                act = get_action(env, obs, info, iters, pgs_original)
                        else:
                            if pid == 0:
                                act = get_action(env, obs, info, iters, pgs_original)
                            else:
                                act = get_action(env, obs, info, iters, pgs_variant)
                        obs, rew, done, info = env.step(act)

                        black_turn = not black_turn

                    black_turn = not black_turn
                    if (black_turn and pid == 0) or (not black_turn and pid == 1):
                        num_victories += rew
            print("Win rate: ", num_victories / (num_games*2))
        pgs_variant.close()
        del pgs_variant