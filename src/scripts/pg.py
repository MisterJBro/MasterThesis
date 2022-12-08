import torch
from multiprocessing import freeze_support
from src.env.hex import HexEnv
from src.networks.residual import HexPolicy
from src.train.config import create_config
from src.train.pg import PPOTrainer


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    freeze_support()

    # Init for algos
    size = 5
    env = HexEnv(size)
    config = create_config({
        "train_iters": 200,
        "env": env,
        "max_len": size*size,
        "num_workers": 3,
        "num_envs": 12,
        "sample_len": 1_000,
        "device": "cuda:0",
        "pi_lr": 6e-4,
        "pi_entropy": 0.0,
        "num_res_blocks": 10,
        "num_filters": 128,
        "ppo_iters": 6,
        "vf_scale": 1.0,
        "clip_ratio": 0.2,
        "sp_num_eval_games": 120,
        "sp_update_win_rate": 0,
        "sp_sampled_policies": 8,
        "use_se": True,
        "use_amp": False,
        "log_main_metric": "win_rate",
        "num_checkpoints": 200,
    })

    # Import policy and model
    policy = HexPolicy(config)

    with PPOTrainer(config, policy) as tr:
        tr.train()
        tr.test()
