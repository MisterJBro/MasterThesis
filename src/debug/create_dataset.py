import torch
import numpy as np
import torch
import random
import numpy as np
from torch.multiprocessing import freeze_support
from src.search.state import State
from src.networks.residual import HexPolicy
from src.search.alpha_zero.alpha_zero import AlphaZero
from src.train.config import create_config
from src.env.hex import HexEnv
from tqdm import trange
from copy import deepcopy
import pickle
import argparse

# import pickle
# with open('epss.pkl', 'rb') as fp:
#   data = pickle.load(fp)
parser = argparse.ArgumentParser(description='File to evaluate methods')

# Parser arguments
parser.add_argument('--job_id', type=int, default=0)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    freeze_support()
    args = parser.parse_args()
    job_id = args.job_id

    # Init for algos
    size = 6
    env = HexEnv(size)
    config = create_config({
        "env": env,
        "puct_c": np.sqrt(2),
        "uct_c": np.sqrt(2),
        "search_num_workers": 3,
        "search_evaluator_batch_size": 3,
        "dirichlet_eps": 0.0,
        "pgs_lr": 1e-1,
        "pgs_trunc_len": 5,
        "device": "cpu",
        "search_return_adv": True,

        "num_res_blocks": 12,
        "num_filters": 128,
        "model_num_res_blocks": 10,
        "model_num_filters": 128,
    })

    # Import policy and model
    policy1 = HexPolicy(config)
    policy1.load("checkpoints/p_6x6_128_12.pt")
    policy1.eval()
    az = AlphaZero(config, policy1)

    def sample_eps(num_games):
        print("SAMPLING:")
        epss = []

        # Simulate
        for _ in trange(num_games):
            obss = []
            acts = []
            vs = []
            states = []
            dists = []
            obs, info = env.reset()

            done = False
            black_turn = True
            while not done:
                states.append(State(deepcopy(env), obs=np.array(obs, copy=True)))
                act = random.choice(np.arange(size**2)[info["legal_act"]])
                next_obs, rew, done, info = env.step(act)

                black_turn = not black_turn
                obss.append(obs)
                acts.append(act)
                obs = next_obs

            black_turn = not black_turn
            obss.append(obs)

            # Get search results
            res = az.search(states, iters=1_000)
            vals = np.concatenate([res["V"], [-rew if black_turn else rew]], 0)
            vs.append(vals)
            dists.append(np.concatenate([res["P"], np.zeros((1, size**2))], 0))

            # Create new eps
            epss.append({
                "obs": np.stack(obss, 0).astype(np.float32),
                "act": np.stack(acts, 0).astype(np.int64),
                "v": np.concatenate(vs).astype(np.float32),
                "pi": np.concatenate(dists).astype(np.float32),
            })
        return epss

    # Sample and save
    eps = sample_eps(1)
    with open(f'eps_{job_id}.pkl', 'wb') as handle:
        pickle.dump(eps, handle, protocol=pickle.HIGHEST_PROTOCOL)

    az.close()