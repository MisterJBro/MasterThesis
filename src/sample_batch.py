from random import sample
import numpy as np

class SampleBatch:
    def __init__(self, num_envs, config):
        self.obs_dtype = config["obs_dtype"]
        self.act_dtype = config["act_dtype"]
        self.rew_dtype = config["rew_dtype"]

        self.idx = 0
        self.obs = np.empty((num_envs, config["sample_len"]) + config["obs_dim"], dtype=self.obs_dtype)
        self.act = np.empty((num_envs, config["sample_len"]), dtype=self.act_dtype)
        self.rew = np.empty((num_envs, config["sample_len"]), dtype=self.rew_dtype)
        self.done = np.empty((num_envs, config["sample_len"]), dtype=np.bool8)
        self.last_obs = np.empty((num_envs,) + config["obs_dim"], dtype=self.obs_dtype)

    def reset(self):
        self.idx = 0

    def set_last_obs(self, obs):
        self.last_obs = obs.astype(self.obs_dtype, copy=False)

    def append(self, obs, act, rew, done):
        self.obs[:, self.idx] = obs.astype(self.obs_dtype, copy=False)
        self.act[:, self.idx] = act.astype(self.act_dtype, copy=False)
        self.rew[:, self.idx] = rew.astype(self.rew_dtype, copy=False)
        self.done[:, self.idx] = done
        self.idx += 1