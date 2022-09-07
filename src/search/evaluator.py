import torch
import time

import numpy as np
from multiprocessing import Process
from multiprocessing.connection import wait


class Evaluator(Process):
    """Evaluation Service for Nodes."""

    def __init__(self, policy, worker_channels, master_channel, device="cpu", batch_size=2, timeout=0.001, use_cache=True):
        Process.__init__(self)
        self.network = policy
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
                    self.update(msg["state_dict"])
                elif msg["command"] == "clear cache":
                    self.clear_cache()

    def eval(self, obs):
        with torch.no_grad():
            dist, val = self.network(obs)
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

        return {
            "wids": wids,
            "reply": reply,
        }

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
        result = self.process(msg)
        wids = result["wids"]
        reply = result["reply"]

        # Send results, no caching yet -> lower latency
        for i, wid in enumerate(wids):
            self.worker_channels[wid].send(reply[i])

        # Cache results:
        if self.use_cache:
            for i, m in enumerate(msg):
                self.cache[m["obs"].tobytes()] = reply[i]

    def update(self, state_dict):
        self.network.load_state_dict(state_dict)


class EvaluatorPGS(Evaluator):
    """Special evaluation Service for network Gradient Search, which returns the hidden states."""

    def eval(self, obs):
        with torch.no_grad():
            pol_h, val_h = self.network.get_hidden(obs)
        res = [{
            "pol_h": p.unsqueeze(0),
            "val_h": v.unsqueeze(0),
        } for p, v in zip(pol_h, val_h)]
        return res

    def update(self, state_dict):
        self.network.load_state_dict(state_dict)
        self.master_channel.send((self.network.network_head, self.network.value_head))


class EvaluatorVE(Evaluator):
    """Evaluator for Value Equivalence Models. Does env step and evaluation together on learned model."""

    def __init__(self, model, worker_channels, master_channel, device="cpu", batch_size=2, timeout=0.001, use_cache=True):
        # Disable cache
        super().__init__(model, worker_channels, master_channel, device, batch_size, timeout, False)

    def process(self, msg):
        # Convert all obs into abstract states (abs)
        reply, obs_list, abs_list = [], [], []

        for m in msg:
            ind = m["ind"]
            obs = m["obs"]
            abs = m["abs"]

            if m["obs"] is not None:
                obs_list.append(obs)
            if m["abs"] is not None:
                abs_list.append(abs)

        if len(obs_list) > 0:
            obs = torch.as_tensor(np.stack(obs_list)).to(self.device)
            with torch.no_grad():
                new_abs = self.network.representation(obs)
            reply.extend([{"abs": a} for a in new_abs])

        obs = np.stack([m["obs"] for m in msg])
        obs = torch.as_tensor(obs).to(self.device)
        wids = [m["ind"] for m in msg]

        return wids, reply