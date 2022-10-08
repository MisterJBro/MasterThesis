from tkinter import E
import numpy as np
from torch.multiprocessing import Pipe
from src.env.worker import Worker


class Envs:
    """Multiprocessing environment class."""

    def __init__(self, config):
        # Get configuration parameter
        self.config = config
        self.num_cpus = config["num_cpus"]
        self.num_envs = config["num_envs"]
        self.device = config["device"]

        # Create rollout worker
        pipes = [Pipe() for _ in range(self.num_cpus)]
        self.channels = [p[0] for p in pipes]
        self.num_envs_worker = int(self.num_envs/self.num_cpus)
        self.rest_env_num = (self.num_envs % self.num_cpus) + self.num_envs_worker
        self.workers = [
            Worker(i, self.rest_env_num if i == self.num_cpus-1 else self.num_envs_worker, pipes[i][1], config)
            for i in range(self.num_cpus)
        ]
        for w in self.workers:
            w.start()

    def reset(self):
        for c in self.channels:
            c.send({"command": "reset"})
        msg = [c.recv() for c in self.channels]
        obs, legal_act = [], []
        for o, la in msg:
            obs.append(o)
            legal_act += la
        obs = np.concatenate(obs)
        return obs, legal_act

    def step(self, act):
        for i, c in enumerate(self.channels):
            if i == self.num_cpus-1:
                c.send({
                    "command": "step",
                    "act": act[i*self.num_envs_worker:],
                })
            else:
                c.send({
                    "command": "step",
                    "act": act[i*self.num_envs_worker:(i+1)*self.num_envs_worker],
                })

        obs_next, rew, done, pid, legal_act = [], [], [], [], []
        msg = [c.recv() for c in self.channels]
        for on, r, d, [p, la] in msg:
            obs_next.append(on)
            rew.append(r)
            done.append(d)
            pid.append(p)
            legal_act += la
        obs_next = np.concatenate(obs_next)
        rew = np.concatenate(rew)
        done = np.concatenate(done)
        pid = np.concatenate(pid)
        info = [pid, legal_act]

        return obs_next, rew, done, info

    def get_all_env(self):
        for c in self.channels:
            c.send({"command": "copy envs"})
        envs = []
        for c in self.channels:
            msg = c.recv()
            envs.extend(msg)
        return envs

    def close(self):
        for c in self.channels:
            c.send({"command": "close"})
        for w in self.workers:
            w.join()
