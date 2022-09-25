import numpy as np
from abc import ABC


class ParallelSearchAlgorithm(ABC):
    """ Interface for search algorithms with several worker threads."""

    def __init__(self, config):
        self.config = config
        self.num_workers = config["search_num_workers"]
        self.num_iters = config["search_iters"]
        self.workers = []
        self.channels = []

    def update(self):
        pass

    def search(self, states, iters=None):
        iters = iters if iters is not None else self.num_iters
        if not isinstance(states, list):
            states = [states]
        i = 0
        dists = []
        num_states = len(states)
        while i < num_states:
            max_c_idx = self.num_workers
            for c_idx, c in enumerate(self.channels):
                c.send({
                    "command": "search",
                    "state": states[i],
                    "iters": iters,
                })
                i += 1
                if i >= num_states:
                    max_c_idx = c_idx+1
                    break
            msg = [c.recv() for c in self.channels[:max_c_idx]]
            dists.extend(msg)
        return np.array(dists)

    def close(self):
        for c in self.channels:
            c.send({"command": "close"})
        for w in self.workers:
            w.join()
