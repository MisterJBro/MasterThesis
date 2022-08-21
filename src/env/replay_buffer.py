import numpy as np
from src.env.sample_batch import SampleBatch


class ReplayBuffer:
    """Replay Buffer for SampleBatches."""

    def __init__(self, config):
        self.config = config
        self.obs_dtype = config["obs_dtype"]
        self.act_dtype = config["act_dtype"]
        self.rew_dtype = config["rew_dtype"]
        self.capacity = config["rep_buf_capacity"]
        self.sample_len = config["sample_len"]
        self.num_envs = config["num_envs"]
        self.device = config["device"]
        size = self.capacity * self.num_envs

        self.idx = 0
        self.obs =  np.empty((size, self.sample_len) + config["obs_dim"], dtype=self.obs_dtype)
        self.act =  np.empty((size, self.sample_len), dtype=self.act_dtype)
        self.rew =  np.empty((size, self.sample_len), dtype=self.rew_dtype)
        self.done = np.empty((size, self.sample_len), dtype=np.bool8)
        self.ret =    np.one((size, self.sample_len), dtype=self.rew_dtype)
        self.val =    np.one((size, self.sample_len), dtype=self.rew_dtype)

    def add(self, sample_batch):
        i1 = self.idx*self.num_envs
        i2 = (self.idx+1)*self.num_envs
        self.obs[i1:i2] = sample_batch.obs
        self.act[i1:i2] = sample_batch.act
        self.rew[i1:i2] = sample_batch.rew
        self.done[i1:i2] = sample_batch.done
        self.ret[i1:i2] = sample_batch.ret
        self.val[i1:i2] = sample_batch.val

        self.idx += 1
        if self.idx == self.capacity:
            self.idx = 0

    def sample(self):
        idx = np.random.choice(self.idx*self.num_envs, self.num_envs, replace=False)

        sample_batch = SampleBatch(self.config)
        sample_batch.obs = self.obs[idx]
        sample_batch.act = self.act[idx]
        sample_batch.rew = self.rew[idx]
        sample_batch.done = self.done[idx]
        sample_batch.ret = self.ret[idx]
        sample_batch.val = self.val[idx]

        return sample_batch

    def __len__(self):
        return self.idx
