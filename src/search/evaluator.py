import torch
import time
import numpy as np
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from multiprocessing.connection import wait


class Evaluator(Process):
    """
    Evaluation Service, manages the policy/networks which typically live on the GPU.
    Bundles requests from workers for more efficient inference.
    """

    def __init__(self, config, policy, worker_channels, master_channel, use_cache=False):
        Process.__init__(self)
        self.policy = policy
        self.worker_channels = worker_channels
        self.master_channel = master_channel
        self.batch_size = config["search_evaluator_batch_size"]
        self.timeout = config["search_evaluator_timeout"]
        self.device = config["device"]
        self.use_cache = use_cache
        self.cache = {}

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

    def process(self, msg):
        obs = np.stack([m["obs"] for m in msg])
        obs = torch.as_tensor(obs).to(self.device)
        wids = [m["ind"] for m in msg]
        reply = self.eval(obs)

        return wids, reply

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

    def update(self, msg):
        self.policy.load_state_dict(msg["policy_params"])
