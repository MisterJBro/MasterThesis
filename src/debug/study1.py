# Naive SP vs delta Uniform SP

import os
import torch
from torch.multiprocessing import freeze_support
from src.train.config import create_config
from src.env.hex import HexEnv
from src.networks.residual import HexPolicy
import numpy as np


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    freeze_support()

    # Init for algos
    size = 7
    env = HexEnv(size)
    config1 = create_config({
        "env": env,
        "device": "cpu",
    })
    config1["num_res_blocks"] = 16
    config1["num_filters"] = 128
    config2 = config1.copy()
    config2["num_res_blocks"] = 16
    config2["num_filters"] = 128

    # Policies
    policy1 = HexPolicy(config1)
    policy2 = HexPolicy(config2)

    # Paths to dirs without ending /
    path1 = "C:/Users/jrb/Desktop/delta_uniform_sp_7x7_128_16_0"
    path2 = "C:/Users/jrb/Desktop/delta_uniform_sp_7x7_128_16_1"

    # Get files
    files1 = [f for f in os.listdir(path1) if os.path.isfile(os.path.join(path1, f))]
    files1 = [f for f in files1 if f.startswith('p')]
    files1.sort(key=lambda f: int(f.split('_')[1]))
    final1 = files1.pop()

    files2 = [f for f in os.listdir(path2) if os.path.isfile(os.path.join(path2, f))]
    files2 = [f for f in files2 if f.startswith('p')]
    files2.sort(key=lambda f: int(f.split('_')[1]))
    final2 = files2.pop()

    # Print number of files
    print(f"Found {len(files1)} files in {path1}")
    print(f"Found {len(files2)} files in {path2}")

    def get_action(env, obs, info, policy):
        # obs (2, 9, 9, 1)
        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(policy.device)
        with torch.no_grad():
            dist, val = policy(obs, legal_actions=info["legal_act"][np.newaxis])
        # Argmax or sample
        #act = dist.logits.argmax(-1).cpu().numpy()[0]
        act = dist.sample().item()
        return act

    # Files 1
    policy_final1 = HexPolicy(config1)
    policy_final1.load(path1 + '/' + final1)
    policy_final1.eval()
    win_rates = []
    for f1 in files1[75:]:
        p1 = path1 + '/' + f1
        # Load
        policy1.load(p1)
        policy1.eval()

        # Simulate
        print(f"{f1} vs {final1}")
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
                            act = get_action(env, obs, info, policy_final1)
                        else:
                            act = get_action(env, obs, info, policy1)
                    else:
                        if pid == 0:
                            act = get_action(env, obs, info, policy1)
                        else:
                            act = get_action(env, obs, info, policy_final1)
                    obs, rew, done, info = env.step(act)

                    black_turn = not black_turn

                black_turn = not black_turn
                if (black_turn and pid == 0) or (not black_turn and pid == 1):
                    num_victories += rew
        print("Win rate: ", num_victories / (num_games*2))
        win_rates.append(num_victories / (num_games*2))

    # Print win rates
    print(f"Win_rates for {path1}: {win_rates}")

    # Files 2
    policy_final2 = HexPolicy(config2)
    policy_final2.load(path2 + '/' + final2)
    policy_final2.eval()
    win_rates = []
    for f2 in files2:
        p2 = path2 + '/' + f2
        # Load
        policy1.load(p2)
        policy1.eval()

        # Simulate
        print(f"{f2} vs {final2}")
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
                            act = get_action(env, obs, info, policy_final2)
                        else:
                            act = get_action(env, obs, info, policy2)
                    else:
                        if pid == 0:
                            act = get_action(env, obs, info, policy2)
                        else:
                            act = get_action(env, obs, info, policy_final2)
                    obs, rew, done, info = env.step(act)

                    black_turn = not black_turn

                black_turn = not black_turn
                if (black_turn and pid == 0) or (not black_turn and pid == 1):
                    num_victories += rew
        print("Win rate: ", num_victories / (num_games*2))
        win_rates.append(num_victories / (num_games*2))

    # Print win rates
    print(f"Win_rates for {path2}: {win_rates}")