import numpy as np
from src.networks.policy_pend import PendulumPolicy
from src.train.trainer import Trainer

class PGTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)
        self.policy = PendulumPolicy(config)

    def update(self, sample_batch):
        data = sample_batch.to_tensor_dict()
        ret = data["ret"]
        val = data["val"]
        adv = ret - val
        data["adv"] = adv

        # Policy and Value loss
        self.policy.loss_gradient(data)
        self.policy.loss_value(data)
