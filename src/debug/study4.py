# PGS vs baselines

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

    # Seed
    seed_all(0)

    # Paths
    path1 = "checkpoints/p_5x5_8_128_weak.pt"

    # Policies
    policy = HexPolicy(config)
    policy.load(path1)
    policy.eval()

    # Algorithms /Players
    def mcts(env, obs, info, iters, sample=True):
        result = mcts_obj.search(State(env, obs=obs), iters=iters)
        pi = result["pi"].reshape(-1)
        if sample:
            act = np.random.choice(len(pi), p=pi)
        else:
            act = np.argmax(pi)
        return act

    def az(env, obs, info, iters, sample=True):
        result = az_obj.search(State(env, obs=obs), iters=iters)
        pi = result["pi"].reshape(-1)
        if sample:
            act = np.random.choice(len(pi), p=pi)
        else:
            act = np.argmax(pi)
        return act

    def mcs(env, obs, info, iters, sample=True):
        result = mcs_obj.search(State(env, obs=obs), iters=iters)
        pi = result["pi"].reshape(-1)
        if sample:
            act = np.random.choice(len(pi), p=pi)
        else:
            act = np.argmax(pi)
        return act

    def pgs_orig(env, obs, info, iters, sample=True):
        result = pgs_obj.search(State(env, obs=obs), iters=iters)
        pi = result["pi"].reshape(-1)
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

    def pn(env, obs, info, iters, sample=True):
        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(policy.device)
        with torch.no_grad():
            dist, val = policy(obs, legal_actions=info["legal_act"][np.newaxis])
        #print("Probs:\n", dist.probs.cpu().numpy().reshape(size, size).round(2))
        if sample:
            act = dist.sample().cpu().item()
        else:
            act = dist.logits.argmax(-1).cpu().numpy()[0]
        return act

    # Simulate
    mcts_obj = None
    az_obj = None
    mcs_obj = None
    pgs_obj = None
    pgs_ext_obj = None
    methods = [mcts, az, mcs, pgs_orig, pgs_ext, pn]
    names = ["mcts", "az", "mcs", "pgs_orig", "pgs_ext", "pn"]
    total_iters = 0
    for i1, name1 in enumerate(names):
        for j1, name2 in enumerate(names):
            # Get method
            if i1 == j1:
                continue

            # Skipper
            if total_iters <= 2:
                total_iters += 1
                continue

            if i1 == 0 or j1 == 0:
                mcts_obj = MCTS(config)
            if i1 == 1 or j1 == 1:
                az_obj = AlphaZero(config, policy)
            if i1 == 2 or j1 == 2:
                mcs_obj = PGS(config, policy, pgs_lr=0e-1)
            if i1 == 3 or j1 == 3:
                pgs_obj = PGS(config, policy, pgs_lr=1e-1)
            if i1 == 4 or j1 == 4:
                pgs_ext_obj = PGS(config, policy, pgs_lr=1e-1, dyn_length=True, scale_vals=True, expl_entr=True, expl_kl=True, visit_counts=True, update=True)

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
                                    act = method1(env, obs, info, iters)
                                else:
                                    act = method2(env, obs, info, iters)
                            else:
                                if pid == 0:
                                    act = method2(env, obs, info, iters)
                                else:
                                    act = method1(env, obs, info, iters)
                            obs, rew, done, info = env.step(act)

                            black_turn = not black_turn

                        black_turn = not black_turn
                        if (black_turn and pid == 0) or (not black_turn and pid == 1):
                            num_victories += rew
                print(f"Iters: {iters} Win rate: {num_victories / (num_games*2)}")

            if i1 == 0 or j1 == 0:
                mcts_obj.close()
                del mcts_obj
            if i1 == 1 or j1 == 1:
                az_obj.close()
                del az_obj
            if i1 == 2 or j1 == 2:
                mcs_obj.close()
                del mcs_obj
            if i1 == 3 or j1 == 3:
                pgs_obj.close()
                del pgs_obj
            if i1 == 4 or j1 == 4:
                pgs_ext_obj.close()
                del pgs_ext_obj
            total_iters  += 1
