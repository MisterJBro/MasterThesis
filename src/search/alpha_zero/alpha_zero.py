from copy import deepcopy
from torch.multiprocessing import Pipe
from src.search.evaluator import Evaluator
from src.search.alpha_zero.worker import AZWorker
from src.search.search import ParallelSearchAlgorithm


class AlphaZero(ParallelSearchAlgorithm):
    """Alpha Zero implementation."""

    def __init__(self, config, policy):
        super().__init__(config)

        # Create workers
        pipes = [Pipe() for _ in range(self.num_workers)]
        eval_pipes = [Pipe() for _ in range(self.num_workers)]
        self.channels = [p[0] for p in pipes]
        self.workers = [AZWorker(config, eval_pipes[i][1], i, pipes[i][1]) for i in range(self.num_workers)]
        for w in self.workers:
            w.start()

        # Create evaluation worker
        eval_master_pipe = Pipe()
        eval_channels = [p[0] for p in eval_pipes]
        self.eval_channel = eval_master_pipe[0]
        self.evaluator = Evaluator(config, deepcopy(policy.cpu()), eval_channels, eval_master_pipe[1])
        self.evaluator.start()

    def update(self, policy_params):
        self.eval_channel.send({
            "command": "update",
            "policy_params": policy_params,
        })

    def search(self, states, iters=None):
        self.eval_channel.send({"command": "clear cache"})
        return super().search(states, iters=iters)

    def close(self):
        for c in self.channels + [self.eval_channel]:
            c.send({"command": "close"})
        for w in self.workers + [self.evaluator]:
            w.join()
