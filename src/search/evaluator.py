import torch
import time

import numpy as np
from multiprocessing import Process
from multiprocessing.connection import wait


class Evaluator(Process):
    """Evaluation Service for Nodes."""

    def __init__(self, policy, worker_channels, master_channel, device="cpu", batch_size=2, timeout=0.001, use_cache=True):
        Process.__init__(self)
        self.policy = policy
        self.worker_channels = worker_channels
        self.master_channel = master_channel
        self.batch_size = batch_size
        self.timeout = timeout
        self.device = device
        self.use_cache = use_cache
        self.cache = {}

    def clear_cache(self):
        self.cache.clear()

    def check_cache(self, reqs):
        msg = []
        for r in reqs:
            m = r.recv()
            if self.use_cache:
                key = m["obs"].tobytes()
                cached = self.cache.get(key, None)
                if cached is not None:
                    self.worker_channels[m["ind"]].send(cached)
                else:
                    msg.append(m)
            else:
                msg.append(m)
        return msg

    def run(self):
        done = False
        while not done:
            if not self.master_channel.poll():
                self.serve_requests()
            else:
                msg = self.master_channel.recv()

                if msg["command"] == "close":
                    done = True
                elif msg["command"] == "update":
                    self.update(msg)
                elif msg["command"] == "clear cache":
                    self.clear_cache()

    def eval(self, obs):
        with torch.no_grad():
            dist, val = self.policy(obs)
        probs = dist.probs.cpu().numpy()
        val = val.cpu().numpy()

        res = [{
            "prob": p,
            "val": v,
        } for p, v in zip(probs, val)]
        return res

    def process(self, msg):
        obs = np.stack([m["obs"] for m in msg])
        obs = torch.as_tensor(obs).to(self.device)
        wids = [m["ind"] for m in msg]
        reply = self.eval(obs)

        return wids, reply

    def serve_requests(self):
        reqs = wait(self.worker_channels, timeout=0.1)

         # Timeout
        if len(reqs) == 0:
            return

        # Wait to fill full batch
        msg = [r.recv() for r in reqs]
        start = time.time()
        while len(msg) < self.batch_size:
            if time.time() - start >= self.timeout:
                break
            reqs = wait(self.worker_channels, timeout=self.timeout/10)
            if len(reqs) == 0:
                continue
            else:
                msg += self.check_cache(reqs)

        # Process the messages
        wids, reply = self.process(msg)

        # Send results, no caching yet -> lower latency
        for i, wid in enumerate(wids):
            self.worker_channels[wid].send(reply[i])

        # Cache results:
        if self.use_cache:
            for i, m in enumerate(msg):
                self.cache[m["obs"].tobytes()] = reply[i]

    def update(self, msg):
        state_dict = msg["state_dict"]
        self.policy.load_state_dict(state_dict)


class PGSEvaluator(Evaluator):
    """Special evaluation Service for network Gradient Search, which returns the hidden states."""

    def eval(self, obs):
        with torch.no_grad():
            pol_h, val_h = self.policy.get_hidden(obs)
        res = [{
            "pol_h": p.unsqueeze(0),
            "val_h": v.unsqueeze(0),
        } for p, v in zip(pol_h, val_h)]
        return res

    def update(self, msg):
        state_dict = msg["state_dict"]
        self.policy.load_state_dict(state_dict)
        self.master_channel.send((self.policy.network_head, self.policy.value_head))


class VEEvaluator(Evaluator):
    """Evaluator for Value Equivalence Models. Does env step and evaluation together on learned model."""

    def __init__(self, policy, model, worker_channels, master_channel, device="cpu", batch_size=2, timeout=0.001, use_cache=True):
        # Disable cache
        super().__init__(policy, worker_channels, master_channel, device, batch_size, timeout, False)
        self.model = model

    def process(self, msg):
        # Convert all obs into abstract states (abs)
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

        if len(obs) > 0:
            obs = torch.as_tensor(np.stack(obs)).to(self.device)
            with torch.no_grad():
                new_abs = self.model.representation(obs)
                new_abs0 = new_abs[0].permute(1, 0, 2)
                new_abs1 = new_abs[1].permute(1, 0, 2)
                dist = self.policy.get_dist(obs)
                prob = dist.probs.cpu().numpy()
            reply.extend([{"abs": (a0.unsqueeze(1), a1.unsqueeze(1)), "prob": p} for a0, a1, p in zip(new_abs0, new_abs1, prob)])

        if len(abs0) > 0 and len(act) == len(abs0):
            abs0 = torch.as_tensor(np.concatenate(abs0, 1)).to(self.device)
            abs1 = torch.as_tensor(np.concatenate(abs1, 1)).to(self.device)
            abs = (abs0, abs1)
            act = torch.as_tensor(np.stack(act)).float().to(self.device)
            with torch.no_grad():
                act = self.model.dyn_linear(act).unsqueeze(1)
                hidden, abs_next = self.model.dynamics(abs, act)
                abs_next0 = abs_next[0].permute(1, 0, 2)
                abs_next1 = abs_next[1].permute(1, 0, 2)
                rew = self.model.get_reward(hidden).cpu().numpy()
                val = self.model.get_value(hidden).cpu().numpy()
                dist = self.model.get_policy(hidden)
                prob = dist.probs.cpu().numpy()
            reply.extend([{
                "abs": (a0.unsqueeze(1), a1.unsqueeze(1)),
                "rew": r,
                "val": v,
                "prob": p,
                } for a0, a1, r, v, p in zip(abs_next0, abs_next1, rew, val, prob)])

        wids = [m["ind"] for m in msg]
        return wids, reply

    def update(self, msg):
        policy_dict = msg["state_dict"]
        model_dict = msg["model_dict"]
        self.policy.load_state_dict(policy_dict)
        self.model.load_state_dict(model_dict)