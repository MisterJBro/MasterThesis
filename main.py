import torch
import numpy as np
from multiprocessing import freeze_support

from src.envs import Envs
from src.config import create_config
from src.policy import ActorCriticPolicy

if __name__ == '__main__':
    freeze_support()
    config = create_config({
        "env": "CartPole-v1",
        "num_cpus": 2,
        "device": "cpu",
        "num_envs": 64,
        "sample_len": 500,
    })

    with Envs(config) as envs:
        policy = ActorCriticPolicy(config)
        params = policy.state_dict()

        import time
        start = time.time()

        sample_batch = envs.sample_batch(params)
        print(sample_batch.obs.shape)

        end = time.time()
        print(f"Time elapsed: {end - start} s")