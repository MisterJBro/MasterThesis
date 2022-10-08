import torch
import numpy as np


class SampleBatch:
    """Batch of sampled data."""

    def __init__(self, config):
        self.obs_dtype = config["obs_dtype"]
        self.act_dtype = config["act_dtype"]
        self.rew_dtype = config["rew_dtype"]
        self.num_envs = config["num_envs"]
        self.device = config["device"]
        self.num_acts = config["num_acts"]

        self.idx = 0
        self.obs = np.empty((self.num_envs, config["sample_len"]) + config["obs_dim"], dtype=self.obs_dtype)
        self.act = np.empty((self.num_envs, config["sample_len"]), dtype=self.act_dtype)
        self.rew = np.empty((self.num_envs, config["sample_len"]), dtype=self.rew_dtype)
        self.done = np.empty((self.num_envs, config["sample_len"]), dtype=np.bool8)
        self.last_obs = np.empty((self.num_envs,) + config["obs_dim"], dtype=self.obs_dtype)
        self.dist = np.empty((self.num_envs, config["sample_len"], self.num_acts), dtype=np.float32)
        self.pid = np.empty((self.num_envs, config["sample_len"]), dtype=np.int8)
        self.ret = None
        self.val = None

    def reset(self):
        self.idx = 0

    def set_last_obs(self, obs):
        self.last_obs = obs.astype(self.obs_dtype, copy=False)

    def append(self, obs, act, rew, done, dist, pid):
        self.obs[:, self.idx] = obs.astype(self.obs_dtype, copy=False)
        self.act[:, self.idx] = act.astype(self.act_dtype, copy=False)
        self.rew[:, self.idx] = rew.astype(self.rew_dtype, copy=False)
        self.done[:, self.idx] = done
        self.dist[:, self.idx] = dist
        self.pid[:, self.idx] = pid

        self.idx += 1

    def get_sections(self):
        sections = []
        batch_len = self.obs.shape[0]
        step_len = self.obs.shape[1]
        for b in range(batch_len):
            start = 0
            for t in range(step_len):
                end = t + 1
                if self.done[b][t]:
                    sections.append((b*step_len + start, b*step_len + end))
                    start = end
            if start < step_len:
                sections.append((b*step_len + start, (b+1)*step_len))
        return sections

    def to_tensor_dict(self):
        obs = torch.from_numpy(self.obs).float()
        obs = obs.flatten(start_dim=0, end_dim=1).to(self.device)
        act = torch.from_numpy(self.act).long().reshape(-1).to(self.device)
        rew = torch.from_numpy(self.rew).float().reshape(-1).to(self.device)
        ret = torch.from_numpy(self.ret).float().reshape(-1).to(self.device)
        val = torch.from_numpy(self.val).float().reshape(-1).to(self.device)
        last_val = torch.from_numpy(self.last_val).float().reshape(-1).to(self.device)
        dist = torch.from_numpy(self.dist).float().reshape(-1, self.num_acts).to(self.device)
        done = torch.from_numpy(self.done).reshape(-1).to(self.device)
        pid = torch.from_numpy(self.pid).reshape(-1).to(self.device)
        sections = self.get_sections()

        return {
            "obs": obs,
            "act": act,
            "rew": rew,
            "ret": ret,
            "val": val,
            "last_val": last_val,
            "dist": dist,
            "done": done,
            "pid": pid,
            "sections": sections,
        }