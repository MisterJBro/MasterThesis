import numpy as np
from copy import deepcopy
from src.search.mcts.worker import TreeWorker
from src.search.search import ParallelSearchAlgorithm
from multiprocessing import Pipe


class MCTS(ParallelSearchAlgorithm):
    """ Monte Carlo Tree Search, with root parallelization."""

    def __init__(self, config):
        super().__init__(config)

        # Create parallel tree workers
        pipes = [Pipe() for _ in range(self.num_workers)]
        self.channels = [p[0] for p in pipes]
        self.workers = [TreeWorker(config, pipes[i][1]) for i in range(self.num_workers)]
        for w in self.workers:
            w.start()

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
