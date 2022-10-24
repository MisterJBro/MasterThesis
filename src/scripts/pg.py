import torch
from multiprocessing import freeze_support
from src.env.hex import HexEnv
from src.networks.residual import HexPolicy
from src.train.config import create_config
from src.train.pg import PGTrainer, PPOTrainer


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    freeze_support()

    # Init for algos
    size = 5
    env = HexEnv(size)
    config = create_config({
        "train_iters": 1_000,
        "env": env,
        "num_cpus": 3,
        "num_envs": 6,
        "sample_len": 2_000,
        "device": "cuda:0",
        "pi_lr": 1e-3,
        "pi_entropy": 0.01,
        "num_batch_split": 30,
        "self_play_num_eval_games": 100,
    })

    # Import policy and model
    policy = HexPolicy(config)

    with PPOTrainer(config, policy) as tr:
        tr.train()
        tr.test()
