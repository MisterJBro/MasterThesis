import torch
from multiprocessing import freeze_support
from src.env.hex import HexEnv
from src.networks.residual import HexPolicy
from src.search.alpha_zero.alpha_zero import AlphaZero
from src.train.config import create_config
from src.train.pg import PGTrainer
from src.train.trainer import Trainer


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
        "search_num_workers": 1,
        "search_evaluator_batch_size": 1,
        "search_return_adv": True,
        "num_cpus": 15,
        "num_envs": 120,
        "device": "cuda:0",
        "pi_lr": 2e-4,
        "vf_lr": 2e-4,
        "pi_entropy": 0.0001,
        "sample_len": 1_000,
    })

    # Import policy and model
    policy = HexPolicy(config)
    #policy.load("checkpoints/policy_hexgame_pg_iter=99_metric=61.pt")
    #model = ValueEquivalenceModel(config)
    #model.load("checkpoints/ve_model.pt")

    #az = AlphaZero(config, policy)

    with PGTrainer(config, policy) as trainer:
        trainer.train()
        trainer.test()
