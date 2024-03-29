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
        reply, obs, act, abs = [], [], [], []
        wids0, wids1 = [], []

        for m in msg:
            if "obs" in m:
                obs.append(m["obs"])
                wids0.append(m["ind"])
            if "abs" in m:
                abs.append(m["abs"])
                wids1.append(m["ind"])
            if "act" in m:
                act.append(m["act"])

        # Evaluate obs and abs
        with torch.no_grad():
            if len(obs) > 0:
                obs = torch.as_tensor(np.stack(obs)).to(self.device)
                reply.extend(self.eval_obs(obs))

            if len(abs) > 0 and len(act) == len(abs):
                abs = torch.as_tensor(np.concatenate(abs, 0)).to(self.device)
                act = torch.as_tensor(act).long().to(self.device)
                reply.extend(self.eval_abs(abs, act))

        wids = wids0 + wids1
        return wids, reply

    def run(self):
        self.model.to(self.model.device)
        while True:
            if not self.master_channel.poll():
                self.serve_requests()
            else:
                msg = self.master_channel.recv()

                if msg["command"] == "close":
                    break
                elif msg["command"] == "update":
                    self.update(msg)
                elif msg["command"] == "clear cache":
                    self.cache.clear()

    def eval_obs(self, obs):
        # Representation network
        new_abs = self.model.representation(obs)
        dist, val = self.model.prediction(new_abs)
        new_abs = new_abs.cpu().numpy()
        prob = dist.probs.cpu().numpy()
        val = val.cpu().numpy()

        return [{
            "abs": a[np.newaxis, :],
            "val": v,
            "prob": p,
            } for a, v, p in zip(new_abs, val, prob)]

    def eval_abs(self, abs, act):
        new_abs, rew = self.model.dynamics(abs.to(self.model.device), act)
        dist, val = self.model.prediction(new_abs)

        # Inference all and then transfer to cpu
        new_abs = new_abs.cpu().numpy()
        rew = rew.cpu().numpy()
        val = val.cpu().numpy()
        prob = dist.probs.cpu().numpy()

        return [{
            "abs": a[np.newaxis, :],
            "rew": r,
            "val": v,
            "prob": p,
            } for a, r, v, p in zip(new_abs, rew, val, prob)]

    def update(self, msg):
        self.model.load_state_dict(msg["model_params"])