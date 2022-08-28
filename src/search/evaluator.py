import numpy as np
import torch
import time

import numpy as np
from multiprocessing import Process
from multiprocessing.connection import wait


class Evaluator(Process):
    """Evaluation Service for Nodes."""

    def __init__(self, policy, worker_channels, master_channel, device="cpu", batch_size=2, timeout=0.001, use_cache=True):
        super().__init__()
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
                    self.update_policy(msg["state_dict"])
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

        # Eval
        obs = np.stack([m["obs"] for m in msg])
        obs = torch.as_tensor(obs).to(self.device)
        inds = [m["ind"] for m in msg]
        res = self.eval(obs)

        # Send results, no caching yet -> lower latency
        for i, ind in enumerate(inds):
            self.worker_channels[ind].send(res[i])

        # Cache results:
        if self.use_cache:
            for i, m in enumerate(msg):
                self.cache[m["obs"].tobytes()] = res[i]

    def update_policy(self, state_dict):
        self.policy.load_state_dict(state_dict)


class EvaluatorPGS(Evaluator):
    """Special evaluation Service for Policy Gradient Search, which returns the hidden states."""

    def eval(self, obs):
        with torch.no_grad():
            pol_h, val_h = self.policy.get_hidden(obs)
        res = [{
            "pol_h": p,
            "val_h": v,
        } for p, v in zip(pol_h, val_h)]
        return res

    def update_policy(self, state_dict):
        self.policy.load_state_dict(state_dict)
        self.master_channel.send((self.policy.policy_head, self.policy.value_head))
