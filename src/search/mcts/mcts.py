import numpy as np
from copy import deepcopy
from src.search.mcts.worker import MCTSWorker
from src.search.search import ParallelSearchAlgorithm
from torch.multiprocessing import Pipe


class MCTS(ParallelSearchAlgorithm):
    """ Monte Carlo Tree Search, with root parallelization."""

    def __init__(self, config):
        super().__init__(config)

        # Create workers
        pipes = [Pipe() for _ in range(self.num_workers)]
        self.channels = [p[0] for p in pipes]
        self.workers = [MCTSWorker(config, pipes[i][1]) for i in range(self.num_workers)]
        for w in self.workers:
            w.start()

    def search(self, state, iters=None):
        iters = iters if iters is not None else self.num_iters
        for c in self.channels:
            c.send({
                "command": "search",
                "state": deepcopy(state),
                "iters": iters,
            })
        msg = [c.recv() for c in self.channels]
        pi = np.mean([m["pi"] for m in msg], axis=0)
        qvals = np.mean([m["q"] for m in msg], axis=0)
        vals = np.mean([m["v"] for m in msg], axis=0)
        return {
            "pi": pi,
            "q": qvals,
            "v": vals,
        }
