from src.search.evaluator import Evaluator
from src.search.mu_zero.evaluator import MZEvaluator


class VEPGSEvaluator(MZEvaluator):
    """Evaluator for PGS. Combining PGS and MuZeros Evaluator"""

    def __init__(self, config, policy, model, worker_channels, master_channel, use_cache=False):
        # Disable cache
        Evaluator.__init__(self, config, policy, worker_channels, master_channel, False)
        self.model = model

    def eval_abs(self, abs, act):
        act = self.model.dyn_linear(act).unsqueeze(1)
        hidden, abs_next = self.model.dynamics(abs, act)
        abs_next0 = abs_next[0].permute(1, 0, 2)
        abs_next1 = abs_next[1].permute(1, 0, 2)

        # Inference only reward
        rew = self.model.get_reward(hidden)
        rew = rew.cpu().numpy()

        return [{
            "abs": (a0.unsqueeze(1), a1.unsqueeze(1)),
            "rew": r,
            "hidden": h,
            } for a0, a1, h, r, in zip(abs_next0, abs_next1, hidden, rew)]

    def update(self, msg):
        self.policy.load_state_dict(msg["policy_params"])
        self.model.load_state_dict(msg["model_params"])
        self.master_channel.send((self.policy.network_head, self.policy.value_head))