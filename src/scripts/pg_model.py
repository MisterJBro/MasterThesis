from src.networks.residual_model import ValueEquivalenceModel
import torch
from multiprocessing import freeze_support
from src.env.hex import HexEnv
from src.networks.residual import HexPolicy
from src.train.config import create_config
from src.train.pg import PGTrainer, PPOTrainer, PPOTrainerModel

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    freeze_support()

    # Init for algos
    size = 6
    env = HexEnv(size)
    config = create_config({
        "train_iters": 500,
        "env": env,
        "max_len": size*size,
        "num_cpus": 3,
        "num_envs": 3,
        "sample_len": 1_000,
        "device": "cuda:0",
        "pi_lr": 6e-4,
        "pi_entropy": 0.0,
        "num_res_blocks": 16,
        "num_filters": 128,
        "ppo_iters": 6,
        "vf_scale": 1.0,
        "clip_ratio": 0.2,
        "num_batch_split": 4,
        "sp_num_eval_games": 100,
        "sp_update_win_rate": 0,
        "use_se": True,
        "log_main_metric": "win_rate",
        "num_checkpoints": 200,

        "model_lr": 1e-3,
        "model_unroll_len": 5,
        "model_minibatches": 50,
        "model_num_res_blocks": 10,
        "model_num_filters": 128,
    })

    # Import policy and model
    policy = HexPolicy(config)
    #policy.load("checkpoints/policy_hex_6x6.pt")
    model = ValueEquivalenceModel(config)

    with PPOTrainerModel(config, policy, model) as tr:
        tr.train()
