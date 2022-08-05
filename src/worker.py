import gym
from enum import Enum, auto
from multiprocessing import Process
import numpy as np


class Command(Enum):
    RESET = auto()
    STEP = auto()
    CLOSE = auto()

class Worker(Process):
    """Environment worker."""

    def __init__(self, idx, num_envs, channel, config):
        super(Worker, self).__init__()
        self.idx = idx
        self.num_envs = num_envs
        self.channel = channel

        # Create environments
        self.envs = [gym.make(config["env"]) for _ in range(num_envs)]
        for i in range(num_envs):
            self.envs[i].seed(i+num_envs*idx + config["seed"])

    def run(self):
        command, acts = self.channel.recv()
        while command != Command.CLOSE:
            if command == Command.RESET:
                self.channel.send(self.reset())
            elif command == Command.STEP:
                self.channel.send(self.step(acts))

            command, acts = self.channel.recv()
        self.close()

    def reset(self):
        obs = np.array([env.reset() for env in self.envs])
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
        obs_next = np.stack(obs_next_list, axis=0)
        rew = np.stack(rew_list, axis=0)
        done = np.stack(done_list, axis=0)

        return obs_next, rew, done

    def close(self):
        for env in self.envs:
            env.close()

