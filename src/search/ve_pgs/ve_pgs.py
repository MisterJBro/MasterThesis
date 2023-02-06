from copy import deepcopy
from torch.multiprocessing import Pipe
from src.search.pgs.pgs import PGS
from src.search.ve_pgs.worker import VEPGSWorker
from src.search.ve_pgs.evaluator import VEPGSEvaluator
from src.search.search import ParallelSearchAlgorithm


class VEPGS(PGS):
    """ Policy Gradient Search with Value Equivalent Model. """

    def __init__(self, config, policy, model, mcs=False, dyn_length=False, scale_vals=False, expl_entr=False, expl_kl=False, visit_counts=False, update=False, puct_c=5.0, trunc_len=5, pgs_lr=1e-1, entr_c=0.1, kl_c=0.1, p_val=0.8):
        ParallelSearchAlgorithm.__init__(self, config)
        self.num_acts = config["num_acts"]

        # Create workers
        pipes = [Pipe() for _ in range(self.num_workers)]
        eval_pipes = [Pipe() for _ in range(self.num_workers)]
        self.channels = [p[0] for p in pipes]
        self.workers = [VEPGSWorker(config, self.num_acts, eval_pipes[i][1], deepcopy(model.pred_pi), deepcopy(model.pred_val), i, pipes[i][1], mcs, dyn_length, scale_vals, expl_entr, expl_kl, visit_counts, update, puct_c, trunc_len, pgs_lr, entr_c, kl_c, p_val) for i in range(self.num_workers)]
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
