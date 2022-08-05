import torch
import numpy as np

from src.envs import Envs
from src.config import create_config
from src.policy import ActorCriticPolicy
from src.process import post_processing
from tabulate import tabulate
from src.sample_batch import SampleBatch
from src.trainer import Trainer

if __name__ == '__main__':
    config = create_config({
        "train_iters": 50,
        "env": "CartPole-v1",
        "num_cpus": 4,
        "device": "cpu",
        "num_envs": 200,
        "sample_len": 500,
        "pi_lr": 1e-3,
        "vf_lr": 8e-4,
    })

    with Trainer(config) as trainer:
        trainer.train()
        trainer.test()
    quit()

    envs = Envs(config)
    print(tabulate([
        ['Environment', config["env"]],
        ['Obs shape', config["obs_dim"]],
        ['Actions num', config["num_acts"]],
        ['CPU count', config["num_cpus"]],
    ], tablefmt="github", colalign=("left", "right")))
    policy = ActorCriticPolicy(config)
    params = policy.state_dict()

    import time
    start = time.time()

    sample_batch = SampleBatch(config["num_envs"], config)
    obs = envs.reset()

    for _ in range(config["sample_len"]):
        act = policy.get_action(obs)
        obs_next, rew, done = envs.step(act)

        sample_batch.append(obs, act, rew, done)
        obs = obs_next

    sample_batch.set_last_obs(obs)
    end = time.time()
    print(f"Time elapsed: {end - start} s") # Old: 990k samples ~8.5s

    sample_batch = post_processing(policy, sample_batch, config)
    print(sample_batch.statistics)