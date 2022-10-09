import torch
from src.train.trainer import Trainer

class PGTrainer(Trainer):
    """ Train a policy using Policy Gradient with baseline."""

    def __init__(self, config, policy):
        super().__init__(config)
        self.policy = policy

    def update(self, sample_batch):
        data = sample_batch.to_tensor_dict()
        ret = data["ret"]
        val = data["val"]
        adv = ret - val
        data["adv"] = adv

        # Policy and Value loss
        self.policy.loss_gradient(data)
        self.policy.loss_value(data)


class PPOTrainer(PGTrainer):
    """ Train a policy using Proximal Policy Gradient."""

    def update(self, sample_batch):
        data = sample_batch.to_tensor_dict()
        ret = data["ret"]
        val = data["val"]
        adv = ret - val
        data["adv"] = adv

        # Policy and Value loss
        self.policy.loss_gradient(data)
        self.policy.loss_value(data)