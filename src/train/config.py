import gym
import torch
import numpy as np
import argparse

# Default configuration of all algorithm. Ideas adopted from ray framework.
DEFAULT_CONFIG = {
    # === Resource settings ===
    "num_cpus": 3,
    "device": "cuda:0",

    # === Environments settings ===
    "env": "CartPole-v1",
    "num_envs": 15,
    "sample_len": 1_000,
    "eval_len": 81,
    "gamma": 1.0,
    "lam": 1.0,
    "seed": 0,
    "test_len": 500,

    "obs_dtype": np.float32,
    "act_dtype": np.float32,
    "rew_dtype": np.float32,

    # === Networks settings ===
    "train_iters": 100,
    "pi_lr": 1e-3,
    "pi_entropy": 0.001,
    "vf_lr": 1e-3,
    "vf_iters": 5,
    "vf_minibatches": 10,
    "model_lr": 1e-3,
    "model_weight_decay": 1e-4,
    "model_iters": 5,
    "model_minibatches": 10,
    "model_unroll_len": 5,
    "grad_clip": 100.0,
    "clip_ratio": 0.2,
    "num_batch_split": 20,
    "num_filters": 128,
    "num_res_blocks": 12,
    "use_se": True,
    "use_amp": False,

    # === Search algorithms ===
    "search_num_workers": 4,
    "search_iters": 1_000,
    "search_evaluator_batch_size": 3,
    "search_evaluator_timeout": 0.001,
    "search_return_adv": True,

    "uct_c": np.sqrt(2),
    "puct_c": 5.0,
    "dirichlet_eps": 0.25,
    "dirichlet_noise": 1.0,
    "pgs_lr": 1e-4,
    "pgs_trunc_len": 10,

    # === Self play ===
    "self_play_elo_k": 30,
    "self_play_num_eval_games": 210,
    "self_play_update_win_rate": 53,

    # === Policy Gradients ===
    "pg_iters": 4,
    "ppo_iters": 10,
    "vf_scale": 1.0,
    "kl_approx_max": 0.1,

    # === Others ===
    "log_main_metric": "elo",
    "log_name": "log.txt",
    "log_to_file": False,
    "log_to_writer": True,
    "num_checkpoints": 1,

}

# Check if configuration is valid, e.g. no illegal parameter values were given like negative learning rate
def check_config(config):
    assert config["num_cpus"] > 0, f'CPU num: {config["num_envs"]} has to be greater 0!'
    assert config["num_envs"] >= config["num_cpus"], f'Env num: {config["num_envs"]} has to be greater or equal to cpu num: {config["num_cpus"]}, so each cpu has atleast one env!'
    assert config["device"] == "cpu" or config["device"].startswith("cuda") and torch.cuda.is_available(), f'Using a device that is not supported: {config["device"]}!'

# Computes missing configuration parameters
def compute_config(config):
    config["num_samples"] = config["sample_len"] * config["num_envs"]

    # Create test env to get obs and act shapes
    if isinstance(config["env"], str):
        env = gym.make(config["env"])
    else:
        env = config["env"]
    config["obs_dim"] = env.observation_space.shape
    config["num_acts"] = env.action_space.n
    config["flat_obs_dim"] = int(np.product(config["obs_dim"]))
    config["num_players"] = env.num_players
    config["amp_device"] = "cpu" if config["device"] == "cpu" else "cuda"

    return config

# Get config from argparser
def args_config(config):
    parser = argparse.ArgumentParser()
    for key, value in config.items():
        if type(value)==bool:
            parser.add_argument(f'--{key}', type=str2bool)
        else:
            parser.add_argument(f'--{key}', type=type(value))

    # Custom arguments
    parser.add_argument('--search_algo', type=str, default="az", choices=['az', 'mz', 'pgs', 'vepgs'])

    args = parser.parse_args()
    args = {k: v for k, v in vars(args).items() if v is not None}
    return args

# From https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Create new config by using own config arguments and the rest from default config
def create_config(new_config):
    config = DEFAULT_CONFIG.copy()
    config.update(new_config)
    config.update(args_config(config))
    config = compute_config(config)
    check_config(config)
    return config
