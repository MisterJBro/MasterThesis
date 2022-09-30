import torch
from src.search.evaluator import Evaluator


class PGSEvaluator(Evaluator):
    """Special evaluation Service for PGS, which returns the hidden states."""

    def eval(self, obs):
        with torch.no_grad():
            pol_h, val_h = self.policy.get_hidden(obs)
        res = [{
            "pol_h": p.unsqueeze(0),
            "val_h": v.unsqueeze(0),
        } for p, v in zip(pol_h, val_h)]
        return res

    def update(self, msg):
        self.policy.load_state_dict(msg["policy_params"])
        self.master_channel.send((self.policy.policy_head, self.policy.value_head))

