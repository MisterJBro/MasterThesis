from multiprocessing import Pipe
import gym
import numpy as np
from src.worker import Worker, Command
from src.sample_batch import SampleBatch


class Envs:
    """Multiprocessing environment class."""

    def __init__(self, config):
        # Get configuration parameter
        self.config = config
        self.num_cpus = config["num_cpus"]
        self.num_envs = config["num_envs"]

        test_env = gym.make(config["env"])
        self.obs_dim = test_env.observation_space.shape
        self.num_acts = test_env.action_space.n
        config["obs_dim"] = self.obs_dim
        config["flat_obs_dim"] = int(np.product(self.obs_dim))
        config["num_acts"] = self.num_acts
        del test_env

        self.channels = [Pipe() for _ in range(self.num_cpus)]
        self.num_envs_worker = int(self.num_envs/self.num_cpus)
        self.rest_env_num = (self.num_envs % self.num_cpus) + self.num_envs_worker
        self.workers = [
            Worker(i, self.rest_env_num if i == self.num_cpus-1 else self.num_envs_worker, self.channels[i][1], config)
            for i in range(self.num_cpus)
        ]
        [w.start() for w in self.workers]

    def sample_batch(self, policy_params):
        for c in self.channels:
            c[0].send([Command.SAMPLE, policy_params])
        sample_batch_list = [c[0].recv() for c in self.channels]

        # Concatenate all samples
        obs = np.concatenate([sb.obs for sb in sample_batch_list], axis=0)
        act = np.concatenate([sb.act for sb in sample_batch_list], axis=0)
        rew = np.concatenate([sb.rew for sb in sample_batch_list], axis=0)
        done = np.concatenate([sb.done for sb in sample_batch_list], axis=0)

        # Create one big sample
        sample_batch = SampleBatch(self.num_envs, self.config)
        sample_batch.obs = obs
        sample_batch.act = act
        sample_batch.rew = rew
        sample_batch.done = done

        return sample_batch

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for c in self.channels:
            c[0].send([Command.CLOSE, None])
        for w in self.workers:
            w.join()
