import torch
import numpy as np

from src.envs import Envs
from src.config import create_config
from src.policy import ActorCriticPolicy
from src.process import post_processing

if __name__ == '__main__':
    config = create_config({
        "env": "CartPole-v1",
        "num_cpus": 1,
        "device": "cpu",
        "num_envs": 16,
        "sample_len": 500,
        "gamma": 0.99,
    })

    with Envs(config) as envs:
        policy = ActorCriticPolicy(config)
        params = policy.state_dict()

        print("Sampled!")
        import time
        start = time.time()

        sample_batch = envs.sample_batch(params)
        sample_batch = post_processing(policy, sample_batch, config)

        end = time.time()
        print(f"Time elapsed: {end - start} s")