import gym
import torch
import numpy as np
import argparse
import pathlib
import yaml


# Default configuration of all algorithm. Ideas adopted from ray framework.
DEFAULT_CONFIG = {
    # === Resource settings ===
    "device": "cuda:0",

    # === Environments settings ===
    "env": "CartPole-v1",
    "num_workers": 3,
    "num_envs": 15,
    "core_pinning": False,
    "sample_len": 1_000,
    "gamma": 1.0,
    "lam": 1.0,
    "seed": 0,
    "max_len": 81,
    "replay_buffer_capacity": 10_000_000,

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
    "grad_clip": 100.0,
    "clip_ratio": 0.2,
    "num_filters": 128,
    "num_res_blocks": 12,
    "batch_size": 2048,
    "acc_grads": 1,
    "use_se": True,
    "use_amp": False,

    # === Model settings ===
    "model_lr": 1e-3,
    "model_weight_decay": 1e-4,
    "model_iters": 5,
    "model_unroll_len": 5,
    "model_num_res_blocks": 10,
    "model_num_filters": 128,
    "model_batch_size": 256,

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
    "sp_elo_k": 30,
    "sp_num_eval_games": 210,
    "sp_update_win_rate": 53,
    "sp_sampled_policies": 4,
    "sp_start_elo": 0,

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
    "log_path": "",
    "num_checkpoints": 1,
    "experiment_path": "",
}

# Check if configuration is valid, e.g. no illegal parameter values were given like negative learning rate
def check_config(config):
    assert config["num_workers"] > 0, f'CPU num: {config["num_envs"]} has to be greater 0!'
    assert config["num_envs"] >= config["num_workers"], f'Env num: {config["num_envs"]} has to be greater or equal to cpu num: {config["num_workers"]}, so each cpu has atleast one env!'
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

    # Envs
    config["num_envs_per_worker"] = config["num_envs"] // config["num_workers"]
    num_envs = config["num_envs_per_worker"] * config["num_workers"]
    if num_envs != config["num_envs"]:
        print(f'Warning: Cannot equally distribute number of of envs: {config["num_envs"]} onto num_workers: {config["num_workers"]}. Setting num_envs to {num_envs}!')
        config["num_envs"] = num_envs
    config["sp_num_eval_games"] = int(np.ceil(config["sp_num_eval_games"]/config["num_envs"])) * config["num_envs"]

    # Paths
    config["root_path"] = pathlib.Path(__file__).parent.parent.parent.absolute().as_posix()
    if config["log_path"] == "":
        config["log_path"] = config["root_path"] + "/src/scripts/log/"
    if config["experiment_path"] == "":
        config["experiment_path"] = config["root_path"] + "/experiments/"

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

def to_yaml(config, path):
    # Do not write "env" field to yaml
    config = config.copy()
    #del config["env"]
    config = {k: str(v) for k, v in config.items()}

    with open(path, "w") as f:
        f.write(yaml.dump(config, default_flow_style=False))
