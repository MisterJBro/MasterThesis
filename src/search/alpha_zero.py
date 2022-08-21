from os import stat
import numpy as np
from copy import deepcopy
from src.search.evaluator import Evaluator
from src.search.tree import AZTreeWorker
from multiprocessing import Pipe


class AlphaZero:
    """ Monte Carlo Tree Search, with root parallelization."""

    def __init__(self, policy, config):
        self.config = config
        self.num_workers = config["num_trees"]
        self.num_iters = config["az_iters"]

        # Create parallel tree workers
        pipes = [Pipe() for _ in range(self.num_workers)]
        eval_pipes = [Pipe() for _ in range(self.num_workers)]
        eval_master_pipe = Pipe()
        self.channels = [p[0] for p in pipes]
        self.eval_channel = eval_master_pipe[0]
        self.num_iters_worker = int(self.num_iters/self.num_workers)
        self.rest_iters = (self.num_iters % self.num_workers) + self.num_iters_worker

        self.workers = []
        for i in range(self.num_workers):
            iters = self.rest_iters if i == self.num_workers-1 else self.num_iters_worker
            eval_pipe = eval_pipes[i][1]

            worker = AZTreeWorker(iters, eval_pipe, i, config, pipes[i][1])
            worker.start()
            self.workers.append(worker)

        # Create evaluation worker
        eval_channels = [p[0] for p in eval_pipes]
        eval_master_channel = eval_master_pipe[1]
        self.eval_worker = Evaluator(policy, eval_channels, eval_master_channel, device=config["device"], batch_size=config["az_eval_batch"], timeout=config["az_eval_timeout"])
        self.eval_worker.start()

    def update_policy(self, state_dict):
        self.eval_channel.send({
            "command": "update",
            "state_dict": state_dict,
        })

    def search(self, state, iters=None):
        for c in self.channels:
            c.send({
                "command": "search",
                "state": deepcopy(state),
                "iters": iters,
            })
        msg = np.stack([c.recv() for c in self.channels])

        qvals = np.mean(msg, axis=0)
        return qvals

    def distributed_search(self, states):
        i = 0
        dists = []
        len_states = len(states)
        while i < len_states:
            max_c_idx = self.num_workers
            for c_idx, c in enumerate(self.channels):
                c.send({
                    "command": "search",
                    "state": states[i],
                    "iters": self.num_iters,
                })
                i += 1
                if i >= len_states:
                    max_c_idx = c_idx+1
                    break
            msg = [c.recv() for c in self.channels[:max_c_idx]]
            dists.extend(msg)
        return np.array(dists)

    def close(self):
        for c in self.channels + [self.eval_channel]:
            c.send({"command": "close"})
        for w in self.workers:
            w.join()
