import torch
from multiprocessing import freeze_support
from src.env.hex import HexEnv
from src.networks.residual import HexPolicy
from src.search.alpha_zero.alpha_zero import AlphaZero
from src.train.config import create_config
from src.train.exit import ExitTrainer


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    freeze_support()

    # Init for algos
    size = 9
    env = HexEnv(size)
    config = create_config({
        "train_iters": 200,
        "env": env,
        "puct_c": 4.0,
        "search_iters": 30,
        "search_return_adv": True,
        "search_num_workers": 3,
        "search_evaluator_batch_size": 3,

        "num_cpus": 3,
        "num_envs": 3,
        "device": "cuda:0",
        "pi_lr": 1e-4,
        "vf_lr": 1e-4,
        "pi_entropy": 0.001,
        "sample_len": 1_0,
        "log_name": f"{'az'}_exit_log.txt",
        "log_to_file": True,
    })

    # Import policy and model
    policy = HexPolicy(config)
    #policy.load("checkpoints/policy_hexgame_pg_iter=99_metric=61.pt")
    #model = ValueEquivalenceModel(config)
    #model.load("checkpoints/ve_model.pt")

    az = AlphaZero(config, policy)

    with ExitTrainer(config, az, policy, model=None) as tr:
        tr.train()
        tr.test()
