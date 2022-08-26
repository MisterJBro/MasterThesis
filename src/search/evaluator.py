import numpy as np
import torch
import time

import numpy as np
from multiprocessing import Process
from multiprocessing.connection import wait


class Evaluator(Process):
    """Evaluation Service for Nodes."""

    def __init__(self, policy, worker_channels, master_channel, device="cpu", batch_size=2, timeout=0.001):
        super().__init__()
        self.policy = policy
        self.worker_channels = worker_channels
        self.master_channel = master_channel
        self.batch_size = batch_size
        self.timeout = timeout
        self.device = device

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

    def eval(self, obs):
        with torch.no_grad():
            dist, val = self.policy(obs)
        probs = dist.probs.cpu().numpy()
        val = val.cpu().numpy()
        return probs, val

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
                msg += [r.recv() for r in reqs]

        # Eval
        obs = np.stack([m["obs"] for m in msg])
        obs = torch.as_tensor(obs).to(self.device)
        inds = [m["ind"] for m in msg]
        probs, val = self.eval(obs)

        # Redistribute res
        for i, ind in enumerate(inds):
            self.worker_channels[ind].send({
                "probs": probs[i],
                "val": val[i],
            })

    def update_policy(self, state_dict):
        self.policy.load_state_dict(state_dict)

class EvaluatorPGS(Evaluator):
    """Special evaluation Service for Policy Gradient Search."""

    def __init__(self, policy, worker_channels, master_channel, device="cpu", batch_size=2, timeout=0.001):
        super().__init__(policy, worker_channels, master_channel, device=device, batch_size=batch_size, timeout=timeout)

    def eval(self, obs):
        with torch.no_grad():
            pol_hidden, val_hidden = self.policy.get_hidden(obs)
        return pol_hidden, val_hidden

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
                msg += [r.recv() for r in reqs]

        # Eval
        obs = np.stack([m["obs"] for m in msg])
        obs = torch.as_tensor(obs).to(self.device)
        inds = [m["ind"] for m in msg]
        pol_hidden, val_hidden = self.eval(obs)

        # Redistribute res
        for i, ind in enumerate(inds):
            self.worker_channels[ind].send({
                "pol_hidden": pol_hidden[i],
                "val_hidden": val_hidden[i],
            })

    def update_policy(self, state_dict):
        self.policy.load_state_dict(state_dict)
        self.master_channel.send((self.policy.policy_head, self.policy.value_head))