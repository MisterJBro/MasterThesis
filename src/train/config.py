import gym
import torch
import numpy as np

# Default configuration of all algorithm. Ideas adopted from ray framework.
DEFAULT_CONFIG = {
    # === Resource settings ===
    "num_cpus": 3,
    "device": "cuda:0",

    # === Environments settings ===
    "env": "CartPole-v1",
    "num_envs": 15,
    "num_players": 1,
    "sample_len": 500,
    "gamma": 1.0,
    "seed": 0,
    "test_len": 500,

    "obs_dtype": np.float32,
    "act_dtype": np.float32,
    "rew_dtype": np.float32,

    # === Models settings ===
    "train_iters": 100,
    "pi_lr": 1e-3,
    "pi_entropy": 0.1,
    "vf_lr": 5e-4,
    "vf_iters": 5,
    "vf_minibatches": 10,
    "model_lr": 5e-4,
    "model_weight_decay": 1e-4,
    "model_iters": 5,
    "model_minibatches": 10,
    "model_unroll_len": 5,
    "grad_clip": 100.0,

    # === MCTS/AlphaZero ===
    "num_trees": 4,
    "uct_c": np.sqrt(2),
    "mcts_iters": 1_000,
    "puct_c": 3.0,
    "az_iters": 1_000,
    "az_eval_batch": 3,
    "az_eval_timeout": 0.001,
    "dirichlet_eps": 0.25,
    "dirichlet_noise": 1.0,

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

    return config

# Create new config by using own config arguments and the rest from default config
def create_config(new_config):
    config = DEFAULT_CONFIG.copy()
    config.update(new_config)
    config = compute_config(config)
    check_config(config)
    return config
