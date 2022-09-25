from copy import deepcopy
from multiprocessing import Pipe
from src.search.pgs.evaluator import PGSEvaluator
from src.search.pgs.worker import PGSWorker
from src.search.search import ParallelSearchAlgorithm


class PGS(ParallelSearchAlgorithm):
    """ Policy Gradient Search with PUCT."""

    def __init__(self, config, policy):
        super().__init__(config)

        # Create parallel tree workers
        pipes = [Pipe() for _ in range(self.num_workers)]
        eval_pipes = [Pipe() for _ in range(self.num_workers)]
        self.channels = [p[0] for p in pipes]
        self.workers = [PGSWorker(config, eval_pipes[i][1], deepcopy(policy.policy_head), deepcopy(policy.value_head), i, pipes[i][1]) for i in range(self.num_workers)]
        for w in self.workers:
            w.start()

        # Create evaluation worker
        eval_master_pipe = Pipe()
        eval_channels = [p[0] for p in eval_pipes]
        self.eval_channel = eval_master_pipe[0]
        self.evaluator = PGSEvaluator(config, policy, eval_channels, eval_master_pipe[1])
        self.evaluator.start()

    def update(self, policy_params):
        self.eval_channel.send({
            "command": "update",
            "policy_params": policy_params,
        })
        pol_head, val_head = self.eval_channel.recv()
        for c in self.channels:
            c.send({
                "command": "update",
                "pol_head": deepcopy(pol_head),
                "val_head": deepcopy(val_head),
            })

    def search(self, states, iters=None):
        self.eval_channel.send({"command": "clear cache"})
        return super().search(states, iters=iters)

    def close(self):
        for c in self.channels + [self.eval_channel]:
            c.send({"command": "close"})
        for w in self.workers + [self.evaluator]:
            w.join()
