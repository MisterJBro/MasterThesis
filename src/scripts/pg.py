import torch
from multiprocessing import freeze_support
from src.env.hex import HexEnv
from src.networks.residual import HexPolicy
from src.train.config import create_config
from src.train.pg import PGTrainer


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    freeze_support()

    # Init for algos
    size = 9
    env = HexEnv(size)
    config = create_config({
        "train_iters": 1_000,
        "env": env,
        "puct_c": 4.0,
        "search_return_adv": True,
        "num_cpus": 15,
        "num_envs": 120,
        "device": "cuda:0",
        "pi_lr": 1e-4,
        "vf_lr": 1e-4,
        "pi_entropy": 0.001,
        "num_batch_split": 30,
        "sample_len": 1_000,
    })

    # Import policy and model
    policy = HexPolicy(config)

    with PGTrainer(config, policy) as tr:
        tr.train()
        tr.test()
