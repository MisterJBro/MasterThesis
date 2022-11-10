import numpy as np
from src.search.evaluator import Evaluator
from src.search.mu_zero.evaluator import MZEvaluator


class VEPGSEvaluator(MZEvaluator):
    """Evaluator for PGS. Combining PGS and MuZeros Evaluator"""

    def __init__(self, config, policy, model, worker_channels, master_channel, use_cache=False):
        # Disable cache
        Evaluator.__init__(self, config, policy, worker_channels, master_channel, False)
        self.model = model

    def eval_abs(self, abs, act):
        new_abs = self.dynamics(abs, act)
        hidden = self.prediction_hidden(new_abs)


        return [{
            "abs": a,
            "rew": np.zeros_like(h),
            "hidden": h,
            } for a, h in zip(abs, hidden)]

    def update(self, msg):
        self.policy.load_state_dict(msg["policy_params"])
        self.model.load_state_dict(msg["model_params"])
        self.master_channel.send((self.policy.network_head, self.policy.value_head))