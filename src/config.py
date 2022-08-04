import torch
import numpy as np

# Default configuration of all algorithm. Ideas adopted from ray framework.
DEFAULT_CONFIG = {
    # === Resource settings ===
    "num_cpus": 3,
    "device": "cuda",

    # === Environments settings ===
    "env": "CartPole-v1",
    "num_envs": 15,
    "sample_len": 500,
    "gamma": 0.99,
    "seed": 0,

    "obs_dtype": np.float32,
    "act_dtype": np.float32,
    "rew_dtype": np.float32,

    # === Models settings ===
    "train_iters": 10,
    "pi_lr": 1e-3,
    "model_lr": 1e-3,
}

# Check if configuration is valid, e.g. no illegal parameter values were given like negative learning rate
def check_config(config):
    assert config["num_cpus"] > 0, f'CPU num: {config["num_envs"]} has to be greater 0!'
    assert config["num_envs"] >= config["num_cpus"], f'Env num: {config["num_envs"]} has to be greater or equal to cpu num: {config["num_cpus"]}, so each cpu has atleast one env!'
    assert config["device"] == "cpu" or config["device"] == "cuda" and torch.cuda.is_available(), f'Using a device that is not supported: {config["device"]}!'


# Create new config by using own config arguments and the rest from default config
def create_config(new_config):
    config = DEFAULT_CONFIG.copy()
    config.update(new_config)
    check_config(config)
    return config
