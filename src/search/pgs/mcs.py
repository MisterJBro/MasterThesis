from copy import deepcopy
from multiprocessing import Pipe
from src.search.pgs.evaluator import PGSEvaluator
from src.search.pgs.pgs import PGS
from src.search.pgs.worker import PGSWorker
from src.search.search import ParallelSearchAlgorithm
from src.train.processer import discount_cumsum


class MCS(PGS):
    """ Monte Carlo Search = PGS without updating the simulation policy. """

    def __init__(self, config, policy):
        ParallelSearchAlgorithm.__init__(self, config)

        # Create parallel tree workers
        pipes = [Pipe() for _ in range(self.num_workers)]
        eval_pipes = [Pipe() for _ in range(self.num_workers)]
        self.channels = [p[0] for p in pipes]
        self.workers = [MCSWorker(config, eval_pipes[i][1], deepcopy(policy.policy_head), deepcopy(policy.value_head), i, pipes[i][1]) for i in range(self.num_workers)]
        for w in self.workers:
            w.start()

        # Create evaluation worker
        eval_master_pipe = Pipe()
        eval_channels = [p[0] for p in eval_pipes]
        self.eval_channel = eval_master_pipe[0]
        self.eval_worker = PGSEvaluator(config, policy, eval_channels, eval_master_pipe[1])
        self.eval_worker.start()


class MCSWorker(PGSWorker):
    """ Worker Process for MCS."""

    def train(self, traj):
        # Only returning the return, no updates
        if len(traj) == 0:
            return 0
        return discount_cumsum(traj["rew"], self.config["gamma"])[0]