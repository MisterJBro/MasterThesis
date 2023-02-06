# VEPGS vs baselines

import random as random_lib
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
from src.search.alpha_zero.alpha_zero import AlphaZero
from src.search.mcts.mcts import MCTS
from src.debug.util import seed_all
from src.networks.residual_model import ValueEquivalenceModel
from src.search.ve_pgs.ve_pgs import VEPGS
from src.search.mu_zero.mu_zero import MuZero


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    freeze_support()

    # Init for algos
    size = 5
    env = HexEnv(size)
    config = create_config({
        "env": env,
        "device": "cpu",
        "uct_c": 5.0,
        "puct_c": 5.0,
        "search_num_workers": 1,
        "search_evaluator_batch_size": 1,
        "dirichlet_eps": 0.0,
        "pgs_lr": 1e-1,
        "pgs_trunc_len": 5,
        "device": "cpu",
        "search_return_adv": False,
    })
    config["num_res_blocks"] = 8
    config["num_filters"] = 128
    config["model_num_res_blocks"] = 4
    config["model_num_filters"] = 128

    # Seed
    seed_all(int(config["job_id"]))

    # Paths
    path_policy = "checkpoints/p_5x5_8_128_weak.pt"
    path_model = "checkpoints/m_5x5_4_128.pt"

    # Model
    model = ValueEquivalenceModel(config)
    model.load(path_model)
    model.eval()

    # Policies
    policy = HexPolicy(config)
    policy.load(path_policy)
    policy.eval()

    # Algorithms /Players
    def mz(env, obs, info, iters, sample=True):
        result = muzero_obj.search(State(env, obs=obs), iters=iters)
        pi = result["pi"].reshape(-1)
        pi = np.exp(pi) / np.sum(np.exp(pi))
        if sample:
            act = np.random.choice(len(pi), p=pi)
        else:
            act = np.argmax(pi)
        return act

    def vepgs(env, obs, info, iters, sample=True):
        result = vepgs_obj.search(State(env, obs=obs), iters=iters)
        pi = result["pi"].reshape(-1)
        pi = np.exp(pi) / np.sum(np.exp(pi))
        if sample:
            act = np.random.choice(len(pi), p=pi)
        else:
            act = np.argmax(pi)
        return act

    def pgs_ext(env, obs, info, iters, sample=True):
        result = pgs_ext_obj.search(State(env, obs=obs), iters=iters)
        pi = result["pi"].reshape(-1)
        pi = np.exp(pi) / np.sum(np.exp(pi))
        if sample:
            act = np.random.choice(len(pi), p=pi)
        else:
            act = np.argmax(pi)
        return act

    def urp(env, obs, info, iters, sample=True):
        return random_lib.choice(np.arange(size**2)[env.legal_actions()])

    # Simulate
    muzero_obj = None
    vepgs_obj = None
    pgs_ext_obj = None
    methods = [mz, vepgs, pgs_ext, urp]
    names = ["mz", "vepgs", "pgs_ext", "urp"]
    total_iters = 0
    for i1, name1 in enumerate(names):
        for j1, name2 in enumerate(names):
            # Get method
            if i1 == j1:
                continue

            if i1 == 0 or j1 == 0:
                muzero_obj = MuZero(config, model)
            if i1 == 1 or j1 == 1:
                vepgs_obj = VEPGS(config, policy, model, pgs_lr=3e-5, dyn_length=False, scale_vals=True, expl_entr=True, expl_kl=True, visit_counts=True, update=True)
            if i1 == 2 or j1 == 2:
                pgs_ext_obj = PGS(config, policy, pgs_lr=3e-5, dyn_length=False, scale_vals=True, expl_entr=True, expl_kl=True, visit_counts=True, update=True)

            method1 = methods[i1]
            method2 = methods[j1]

            print(f"Testing: {name1} vs {name2}")

            for iters in [50, 100, 200]:
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
                                    act = method1(env, obs, info, iters, sample=False)
                                else:
                                    act = method2(env, obs, info, iters, sample=False)
                            else:
                                if pid == 0:
                                    act = method2(env, obs, info, iters, sample=False)
                                else:
                                    act = method1(env, obs, info, iters, sample=False)
                            obs, rew, done, info = env.step(act)

                            black_turn = not black_turn

                        black_turn = not black_turn
                        if (black_turn and pid == 0) or (not black_turn and pid == 1):
                            num_victories += rew
                print(f"Iters: {iters} Win rate: {num_victories / (num_games*2)}")

            if i1 == 0 or j1 == 0:
                muzero_obj.close()
                del muzero_obj
            if i1 == 1 or j1 == 1:
                vepgs_obj.close()
                del vepgs_obj
            if i1 == 2 or j1 ==2:
                pgs_ext_obj.close()
                del pgs_ext_obj
            total_iters  += 1
