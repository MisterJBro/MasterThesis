import gym
import numpy as np
import os

from enum import Enum, auto
from src.policy import ActorCriticPolicy
from src.sample_batch import SampleBatch
from multiprocessing import Process, Queue, Pipe
from multiprocessing import freeze_support


class Command(Enum):
    SAMPLE = auto()
    CLOSE = auto()

class Worker(Process):
    """Environment worker."""

    def __init__(self, idx, num_envs, channel, config):
        Process.__init__(self)
        os.environ["MKL_NUM_THREADS"] = "1"
        self.idx = idx
        self.num_envs = num_envs
        self.channel = channel
        self.env = config["env"]
        self.sample_len = config["sample_len"]
        self.obs_dim = config["obs_dim"]
        self.behavior_policy = ActorCriticPolicy(config)
        self.behavior_policy.eval()

        # Create environments
        self.envs = [gym.make(self.env) for _ in range(num_envs)]
        for i in range(num_envs):
            self.envs[i].seed(i+num_envs*idx)

        # Sampled data
        self.sample_batch = SampleBatch(num_envs, config)

    def run(self):
        command, params = self.channel.recv()
        while command != Command.CLOSE:
            if command == Command.SAMPLE:
                self.channel.send(self.get_sample_batch(params))
            command, params = self.channel.recv()

        self.close()

    def get_sample_batch(self, policy_params):
        self.behavior_policy.load_state_dict(policy_params)

        obs = self.reset()
        for _ in range(self.sample_len):
            act = self.behavior_policy.get_action(obs)
            obs_next, rew, done = self.step(act)

            self.sample_batch.append(obs, act, rew, done)
            obs = obs_next
        return self.sample_batch

    def reset(self):
        obs = np.array([env.reset()
                         for env in self.envs])
        return obs

    def step(self, acts):
        obs_next_list, rew_list, done_list = [], [], []

        for i in range(self.num_envs):
            obs_next, rew, done, _ = self.envs[i].step(acts[i])

            if done:
                obs_next = self.envs[i].reset()

            obs_next_list.append(obs_next)
            rew_list.append(rew)
            done_list.append(done)
        obs_next, rew, done = np.array(obs_next_list), np.array(rew_list), np.array(done_list)

        return obs_next, rew, done

    def close(self):
        for env in self.envs:
            env.close()

