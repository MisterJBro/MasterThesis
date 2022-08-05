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
        self.ret = None
        self.val = None

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

    def get_episodes(self):
        episodes = []
        for b in range(self.obs.shape[0]):
            start = 0
            for t in range(self.obs.shape[1]):
                end = t + 1
                if self.done[b][t]:
                    episodes.append((self.obs[b][start:end], self.act[b][start:end], self.rew[b][start:end], self.ret[b][start:end], self.val[b][start:end], self.last_obs[b]))
                    start = end
            if start < self.obs.shape[1]:
                episodes.append((self.obs[b][start:], self.act[b][start:], self.rew[b][start:], self.ret[b][start:], self.val[b][start:], self.last_obs[b]))
        return episodes