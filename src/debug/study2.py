# Policy vs MCTS different number of iterations

import os
import torch
from torch.multiprocessing import freeze_support
from src.train.config import create_config
from src.env.hex import HexEnv
from src.networks.residual import HexPolicy
from src.search.mcts.mcts import MCTS
import numpy as np
import math
from src.search.state import State


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    freeze_support()

    # Init for algos
    size = 7
    env = HexEnv(size)
    config = create_config({
        "env": env,
        "device": "cpu",
        "uct_c": math.sqrt(2),
        "search_num_workers": 1,
        "search_return_adv": False,
    })
    config["num_res_blocks"] = 16
    config["num_filters"] = 128
    mcts_obj = MCTS(config)

    # Paths
    path1 = "C:/Users/jrb/Desktop/delta_uniform_sp_7x7_128_16_0/p_100_elo_2255.6.pt"

    # Policies
    policy1 = HexPolicy(config)
    policy1.load(path1)
    policy1.eval()

    # Algorithms /Players
    def mcts(env, obs, info, iters):
        result = mcts_obj.search(State(env, obs=obs), iters=iters)
        #act = np.argmax(result["pi"])
        # Softmax of pi
        pi = np.exp(result["pi"])/sum(np.exp(result["pi"]))
        act = np.random.choice(np.arange(len(pi)), p=pi)
        return act

    def get_action(env, obs, info, policy):
        # obs (2, 9, 9, 1)
        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(policy.device)
        with torch.no_grad():
            dist, val = policy(obs, legal_actions=info["legal_act"][np.newaxis])
        # Argmax or sample
        #act = dist.logits.argmax(-1).cpu().numpy()[0]
        act = dist.sample().item()
        return act

    # Simulate
    for iters in [50, 100, 200, 300, 400, 500, 600]:
        print("Iters: ", iters)
        num_victories = 0
        num_games = 100
        for pid in range(2):
            for i in range(num_games):
                obs, info = env.reset()

                done = False
                black_turn = True
                while not done:
                    # Action
                    if black_turn:
                        if pid == 0:
                            act = get_action(env, obs, info, policy1)
                        else:
                            act = mcts(env, obs, info, iters)
                    else:
                        if pid == 0:
                            act = mcts(env, obs, info, iters)
                        else:
                            act = get_action(env, obs, info, policy1)
                    obs, rew, done, info = env.step(act)

                    black_turn = not black_turn

                black_turn = not black_turn
                if (black_turn and pid == 0) or (not black_turn and pid == 1):
                    num_victories += rew
        print("Win rate: ", num_victories / (num_games*2))