import numpy as np
from multiprocessing import Pipe
from src.worker import Worker, Command


class Envs:
    """Multiprocessing environment class."""

    def __init__(self, config):
        # Get configuration parameter
        self.config = config
        self.num_cpus = config["num_cpus"]
        self.num_envs = config["num_envs"]

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
            c.send([Command.RESET, None])
        msg = [c.recv() for c in self.channels]

        self.obs = np.concatenate(msg)
        return self.obs

    def step(self, act):
        for i, c in enumerate(self.channels):
            if i == self.num_cpus-1:
                c.send([Command.STEP, act[i*self.num_envs_worker:]])
            else:
                c.send([Command.STEP, act[i*self.num_envs_worker:(i+1)*self.num_envs_worker]])

        obs_next, rew, done = [], [], []
        msg = [c.recv() for c in self.channels]
        for on, r, d in msg:
            obs_next.append(on)
            rew.append(r)
            done.append(d)
        obs_next = np.concatenate(obs_next, axis=0)
        rew = np.concatenate(rew, axis=0)
        done = np.concatenate(done, axis=0)

        return obs_next, rew, done

    def close(self):
        for c in self.channels:
            c[0].send([Command.CLOSE, None])
        for w in self.workers:
            w.join()
