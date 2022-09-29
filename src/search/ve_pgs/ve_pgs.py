from copy import deepcopy
from torch.multiprocessing import Pipe
from src.search.pgs.pgs import PGS
from src.search.ve_pgs.worker import VEPGSWorker
from src.search.ve_pgs.evaluator import VEPGSEvaluator
from src.search.search import ParallelSearchAlgorithm


class VEPGS(PGS):
    """ Policy Gradient Search with Value Equivalent Model. """

    def __init__(self, config, policy, model):
        ParallelSearchAlgorithm.__init__(self, config)
        self.num_acts = config["num_acts"]

        # Create workers
        pipes = [Pipe() for _ in range(self.num_workers)]
        eval_pipes = [Pipe() for _ in range(self.num_workers)]
        self.channels = [p[0] for p in pipes]
        self.workers = [VEPGSWorker(config, self.num_acts, eval_pipes[i][1], deepcopy(model.pre_pi), deepcopy(model.pre_val), i, pipes[i][1]) for i in range(self.num_workers)]
        for w in self.workers:
            w.start()

        # Create evaluation worker
        eval_master_pipe = Pipe()
        eval_channels = [p[0] for p in eval_pipes]
        self.eval_channel = eval_master_pipe[0]
        self.evaluator = VEPGSEvaluator(config, policy, model, eval_channels, eval_master_pipe[1])
        self.evaluator.start()

    def update(self, policy_params, model_params):
        self.eval_channel.send({
            "command": "update",
            "policy_params": policy_params,
            "model_params": model_params,
        })
        pol_head, val_head = self.eval_channel.recv()
        for c in self.channels:
            c.send({
                "command": "update",
                "pol_head": deepcopy(pol_head),
                "val_head": deepcopy(val_head),
            })
