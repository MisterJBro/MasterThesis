import torch
from multiprocessing import freeze_support
from src.env.hex import HexEnv
from src.networks.residual import HexPolicy
from src.networks.model import ValueEquivalenceModel
from src.search.alpha_zero.alpha_zero import AlphaZero
from src.search.mu_zero.mu_zero import MuZero
from src.search.pgs.pgs import PGS
from src.search.ve_pgs.ve_pgs import VEPGS
from src.train.config import create_config
from src.train.exit import ExitTrainer


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    freeze_support()

    # Init for algos
    size = 5
    env = HexEnv(size)
    config = create_config({
        "env": env,
        "puct_c": 4.0,
        "train_iters": 50,
        "search_iters": 100,
        "search_num_workers": 3,
        "search_evaluator_batch_size": 3,
        "num_cpus": 3,
        "num_envs": 3,
        "device": "cuda:0",
        "pi_lr": 1e-3,
        "sample_len": 100,
        "search_return_adv": True,
        "log_name": f"exit_log.txt",
    })

    # Import policy and model
    policy = HexPolicy(config)
    model = None

    # Algorithms
    search_algo = config["search_algo"]
    if search_algo == "az":
        search_algo = AlphaZero(config, policy)
    elif search_algo == "mz":
        model = ValueEquivalenceModel(config)
        search_algo = MuZero(config, policy, model)
    elif search_algo == "pgs":
        search_algo = PGS(config, policy)
    elif search_algo == "vepgs":
        model = ValueEquivalenceModel(config)
        search_algo = VEPGS(config, policy, model)
    else:
        raise ValueError('Unknown search algorithm: ', search_algo)

    # Train
    with ExitTrainer(config, search_algo, policy, model=model) as tr:
        tr.train()
        tr.save()
