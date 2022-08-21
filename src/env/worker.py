import gym
import numpy as np
from copy import deepcopy
from multiprocessing import Process


class Worker(Process):
    """Environment worker."""

    def __init__(self, idx, num_envs, channel, config):
        super(Worker, self).__init__()
        self.idx = idx
        self.num_envs = num_envs
        self.channel = channel

        # Create environments
        self.envs = []
        for _ in range(num_envs):
            if isinstance(config["env"], str):
                self.envs.append(gym.make(config["env"]))
            else:
                self.envs.append(deepcopy(config["env"]))
        for i in range(num_envs):
            self.envs[i].seed(i+num_envs*idx + config["seed"])

    def run(self):
        while True:
            msg = self.channel.recv()
            if msg["command"] == "step":
                self.channel.send(self.step(msg["act"]))
            elif msg["command"] == "reset":
                self.channel.send(self.reset())
            elif msg["command"] == "copy envs":
                self.channel.send(deepcopy(self.envs))
            elif msg["command"] == "close":
                break

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

