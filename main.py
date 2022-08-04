import torch
import numpy as np

from src.envs import Envs
from src.config import create_config
from src.policy import ActorCriticPolicy
from src.process import post_processing
from tabulate import tabulate

if __name__ == '__main__':
    config = create_config({
        "train_iters": 10,
        "env": "CartPole-v1",
        "num_cpus": 2,
        "device": "cpu",
        "num_envs": 2,
        "sample_len": 30,
        "gamma": 0.99,
    })

    envs = Envs(config)
    print(tabulate([
        ['Environment', config["env"]],
        ['Obs shape', config["obs_dim"]],
        ['Actions num', config["num_acts"]],
        ['CPU count', config["num_cpus"]],
    ], tablefmt="github", colalign=("left", "right")))
    policy = ActorCriticPolicy(config)
    params = policy.state_dict()

    print("Sampled!")
    import time
    start = time.time()

    sample_batch = envs.sample_batch(params)
    sample_batch = post_processing(policy, sample_batch, config)
    print(sample_batch.statistics)

    end = time.time()
    print(f"Time elapsed: {end - start} s")