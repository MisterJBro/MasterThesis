import torch
import numpy as np
from src.search.evaluator import Evaluator


class MZEvaluator(Evaluator):
    """Evaluator for MuZero. Does env step and evaluation together on learned model."""

    def __init__(self, config, policy, model, worker_channels, master_channel, use_cache=False):
        # Disable cache
        super().__init__(config, policy, worker_channels, master_channel, False)
        self.model = model

    def process(self, msg):
        # Filter out all obs and abs from the messages
        reply, obs, act = [], [], []
        abs0, abs1 = [], []

        for m in msg:
            if "obs" in m:
                obs.append(m["obs"])
            if "abs" in m:
                abs0.append(m["abs"][0])
                abs1.append(m["abs"][1])
            if "act" in m:
                act.append(m["act"])

        # Evaluate obs and abs
        with torch.no_grad():
            if len(obs) > 0:
                obs = torch.as_tensor(np.stack(obs)).to(self.device)
                reply.extend(self.eval_obs(obs))

            if len(abs0) > 0 and len(act) == len(abs0):
                abs0 = torch.as_tensor(np.concatenate(abs0, 1)).to(self.device)
                abs1 = torch.as_tensor(np.concatenate(abs1, 1)).to(self.device)
                abs = (abs0, abs1)
                act = torch.as_tensor(np.stack(act)).float().to(self.device)
                reply.extend(self.eval_abs(abs, act))

        wids = [m["ind"] for m in msg]
        return wids, reply

    def eval_obs(self, obs):
        # Representation network
        new_abs = self.model.representation(obs)
        new_abs0 = new_abs[0].permute(1, 0, 2)
        new_abs1 = new_abs[1].permute(1, 0, 2)
        dist = self.policy.get_dist(obs)
        prob = dist.probs.cpu().numpy()

        return [{"abs": (a0.unsqueeze(1), a1.unsqueeze(1)), "prob": p} for a0, a1, p in zip(new_abs0, new_abs1, prob)]

    def eval_abs(self, abs, act):
        act = self.model.dyn_linear(act).unsqueeze(1)
        hidden, abs_next = self.model.dynamics(abs, act)
        abs_next0 = abs_next[0].permute(1, 0, 2)
        abs_next1 = abs_next[1].permute(1, 0, 2)

        rew = self.model.get_reward(hidden).cpu().numpy()
        val = self.model.get_value(hidden).cpu().numpy()
        dist = self.model.get_policy(hidden)
        prob = dist.probs.cpu().numpy()

        return [{
            "abs": (a0.unsqueeze(1), a1.unsqueeze(1)),
            "rew": r,
            "val": v,
            "prob": p,
            } for a0, a1, r, v, p in zip(abs_next0, abs_next1, rew, val, prob)]

    def update(self, msg):
        self.policy.load_state_dict(msg["policy_params"])
        self.model.load_state_dict(msg["model_params"])