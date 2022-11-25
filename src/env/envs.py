import numpy as np
from torch.multiprocessing import Pipe
from src.env.worker import Worker


class Envs:
    """Multiprocessing environment class."""

    def __init__(self, config):
        # Get configuration parameter
        self.config = config
        self.num_workers = config["num_workers"]
        self.num_envs = config["num_envs"]
        self.device = config["device"]

        # Create rollout worker
        pipes = [Pipe() for _ in range(self.num_workers)]
        self.channels = [p[0] for p in pipes]
        self.num_envs_worker = int(self.num_envs/self.num_workers)
        self.rest_env_num = (self.num_envs % self.num_workers) + self.num_envs_worker
        self.workers = [
            Worker(i, self.rest_env_num if i == self.num_workers-1 else self.num_envs_worker, pipes[i][1], config)
            for i in range(self.num_workers)
        ]
        for w in self.workers:
            w.start()

    def reset(self):
        for c in self.channels:
            c.send({"command": "reset"})
        msg = [c.recv() for c in self.channels]
        obs, legal_act, pid = [], [], []
        for o, i in msg:
            obs.append(o)
            legal_act += i["legal_act"]
            pid += i["pid"]
        obs = np.concatenate(obs)
        legal_act = np.stack(legal_act)
        return obs, {"legal_act": legal_act, "pid": pid}

    def step(self, act):
        for i, c in enumerate(self.channels):
            if i == self.num_workers-1:
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
        for on, r, d, i in msg:
            obs_next.append(on)
            rew.append(r)
            done.append(d)

            pid += i["pid"]
            legal_act += i["legal_act"]
        obs_next = np.concatenate(obs_next)
        rew = np.concatenate(rew)
        done = np.concatenate(done)
        legal_act = np.stack(legal_act)
        info = {"pid": pid, "legal_act": legal_act}

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
