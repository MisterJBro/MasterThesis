import numpy as np
from copy import deepcopy
from multiprocessing import Pipe
from src.search.evaluator import PGSEvaluator
from src.search.pgs.worker import PGSTreeWorker


class PGS:
    """ Policy Gradient Search with PUCT."""

    def __init__(self, policy, config):
        self.config = config
        self.num_workers = config["num_trees"]
        self.num_iters = config["pgs_iters"]

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

            pol_head = deepcopy(policy.policy_head)
            val_head = deepcopy(policy.value_head)
            worker = PGSTreeWorker(iters, eval_pipe, pol_head, val_head, i, config, pipes[i][1])
            worker.start()
            self.workers.append(worker)

        # Create evaluation worker
        eval_channels = [p[0] for p in eval_pipes]
        eval_master_channel = eval_master_pipe[1]
        self.eval_worker = PGSEvaluator(policy, eval_channels, eval_master_channel, device=config["device"], batch_size=config["pgs_eval_batch"], timeout=config["pgs_eval_timeout"])
        self.eval_worker.start()

    def update_policy(self, state_dict):
        self.eval_channel.send({
            "command": "update",
            "state_dict": state_dict,
        })
        pol_head, val_head = self.eval_channel.recv()
        for c in self.channels:
            c.send({
                "command": "update",
                "pol_head": deepcopy(pol_head),
                "val_head": deepcopy(val_head),
            })

    def search(self, state, iters=None):
        self.eval_channel.send({"command": "clear cache"})
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
        self.eval_channel.send({"command": "clear cache"})
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
        for w in self.workers + [self.eval_worker]:
            w.join()
