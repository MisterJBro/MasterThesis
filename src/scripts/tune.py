from hyperopt import hp, tpe, fmin, Trials
import torch
import numpy as np
from multiprocessing import freeze_support
from src.env.hex import HexEnv
from src.networks.residual import HexPolicy
from src.train.config import create_config
from src.train.pg import PGTrainer, PPOTrainer


# Objective
def objective(params):
    print(f"Testing out params: {params}")

    num_batch_split = 25
    if params["sample_len"] == 2000:
        num_batch_split = 50
    elif params["sample_len"] == 4000:
        num_batch_split = 75

    # Init for algos
    size = 9
    env = HexEnv(size)
    config = {
        "train_iters": 20,
        "log_main_metric": "win_rate",
        "env": env,
        "num_cpus": 15,
        "num_envs": 120,
        "device": "cuda:0",
        "self_play_num_eval_games": 240,
        "num_batch_split": num_batch_split,
        "grad_clip": 100.0,
        "log_to_writer": False,
        "log_to_file": True,
    }
    config.update(params)
    config = create_config(config)

    # Import policy and model
    policy = HexPolicy(config)

    with PPOTrainer(config, policy) as tr:
        tr.train()
        win_rates = np.array(tr.log.main_metric)

    # Get metrics
    max_wr = np.max(win_rates)
    x = np.arange(len(win_rates))
    A = np.vstack([x, np.ones(len(x))]).T
    slope_wr, _ = np.linalg.lstsq(A, win_rates, rcond=None)[0]
    max_drop_wr = np.max(win_rates[:-1] - win_rates[1:])

    # Write progress
    f = open("tune_log.txt", "a")
    f.write(f'Max Winrate: {max_wr:.02f}  Slope: {slope_wr:.02f}  Max Drop: {int(max_drop_wr)}\n\tWinrates: {np.round(win_rates, 1)}\n\tParams: {params}\n')
    f.close()

    return -(max_wr - 0.5*max_drop_wr) * (1 if slope_wr > 0 else 0)

# Space
space = {
    "pi_lr": hp.choice("pi_lr", [6e-3, 4e-3, 3e-3, 6e-5]),
    "pi_entropy": hp.uniform("pi_entropy", 0.01, 0.02),
    "clip_ratio": hp.uniform("clip_ratio", 0.1, 0.3),
    "num_filters": hp.choice("num_filters", [128, 192]),
    "num_res_blocks": hp.choice("num_res_blocks", [12, 16, 18]),
    "use_se": hp.choice("use_se", [False, True]),
    "ppo_iters": hp.randint("ppo_iters", 10) + 10,
    "vf_scale": hp.uniform("vf_scale", 0.6, 1.4),
    "kl_approx_max": hp.uniform("kl_approx_max", 0.5, 0.8),
    "sample_len": hp.choice("sample_len", [1000]),
}

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    freeze_support()

    # Optimizing 
    trials = Trials()
    best = fmin(objective, space=space, algo=tpe.suggest, max_evals=200, trials=trials)

    # Print results
    print(best)
    print(trials.results)
