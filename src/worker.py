import gym
import numpy as np
import ray
import os

from src.sample_batch import SampleBatch


@ray.remote(num_cpus=1, num_gpus=0)
class Worker(object):
    """Environment worker."""

    def __init__(self, idx, env, num_envs, config, PolicyClass):
        super(Worker, self).__init__()
        os.environ["MKL_NUM_THREADS"] = "1"
        self.idx = idx
        self.env = env
        self.num_envs = num_envs
        self.sample_len = config["sample_len"]
        self.obs_dim = config["obs_dim"]
        self.behavior_policy = PolicyClass()

        # Create environments
        self.envs = [gym.make(env) for _ in range(num_envs)]
        for i in range(num_envs):
            self.envs[i].seed(i+num_envs*idx)

        # Sampled data
        self.sample_batch = SampleBatch(num_envs, config)

    def sample_batch(self):
        obs = self.reset()
        for _ in range(self.sample_len):
            act = self.behavior_policy.get_action(obs)
            obs_next, rew, done = self.step(act)
            self.sample_batch.append(obs, act, rew, done)

            obs = obs_next

    def reset(self):
        obs = np.array([env.reset()
                         for env in self.envs])
        return obs

    def step(self, acts):
        obs_next_list, rew_list, done_list = [], [], []

        for i in range(self.num_envs):
            n_obs, rew, done, _ = self.envs[i].step(acts[i])

            if done:
                n_obs = self.envs[i].reset()

            obs_next_list.append(n_obs)
            rew_list.append(rew)
            done_list.append(done)
        n_obs, rew, done = np.array(obs_next_list), np.array(rew_list), np.array(done_list)

        return n_obs, rew, done

    def close(self):
        for env in self.envs:
            env.close()
