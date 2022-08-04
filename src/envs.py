import ray
import gym
from gym.spaces import Box
import numpy as np
from multiprocessing import Pipe
from .buffer import Buffer
from .utils import to_tensors
from .worker import Worker


class Envs:
    """Multiprocessing environment class."""

    def __init__(self, config):
        # Get configuration parameter
        self.num_cpus = config["num_cpus"]
        self.num_envs = config["num_envs"]

        test_env = gym.make(config["env"])
        self.obs_dim = test_env.observation_space.shape
        self.num_acts = test_env.action_space.n
        config["obs_dim"] = self.obs_dim
        config["num_acts"] = self.num_acts
        del test_env

        self.num_envs_worker = int(self.num_envs/self.num_cpus)
        self.rest_env_num = (self.num_envs % self.num_cpus) + self.num_envs_worker
        self.workers = [
            Worker(i, config["env"], self.rest_env_num if i == self.num_cpus-1 else self.num_envs_worker, config, )
                .options(name=f"RolloutWorker {i}")
                .remote()
            for i in range(self.num_cpus)
        ]
        self.obss = None

    def reset(self):
        obs = ray.get([w.reset.remote() for w in self.workers])
        return obs

    def step(self, acts, vals):
        [c[0].send(['step', acts[i*self.num_envs_worker:self.num_envs if i == self.num_cpus-1 else (i+1)*self.num_envs_worker]])
         for i, c in enumerate(self.channels)]
        msg = [c[0].recv() for c in self.channels]
        obs_msg, rew_msg = [], []
        for i, (o, r, d) in enumerate(msg):
            obs_msg.append(o)
            rew_msg.append(r)

            for j in range(self.num_envs_worker):
                if d[j]:
                    index = j + self.num_envs_worker*i
                    self.buf.sections[index].append(self.buf.ptr+1)

        rews = np.concatenate(rew_msg, axis=0)
        n_obss = np.concatenate(obs_msg, axis=0)
        self.buf.store(self.obss, acts, rews, vals)
        self.obss = n_obss

        return n_obss

    def ret_and_adv(self):
        self.buf.ret_and_adv()
        return self.buf.avg_rew()

    def get_data(self):
        return to_tensors(self.buf.get_data())

    def close(self):
        [c[0].send(['close', None]) for c in self.channels]
