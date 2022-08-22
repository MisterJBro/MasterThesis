import numpy as np
from src.networks.policy_pend import PendulumPolicy
from src.train.trainer import Trainer

class PGTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)
        self.policy = PendulumPolicy(config)

    def train(self):
        for iter in range(self.config["train_iters"]):
            sample_batch = self.get_sample_batch()
            self.update(sample_batch)
            stats = sample_batch.statistics

            avg_ret = stats["mean_return"]
            max_ret = stats["max_return"]
            min_ret = stats["min_return"]
            print(f'Iteration: {iter}  Avg Ret: {np.round(avg_ret, 3)}  Max Ret: {np.round(max_ret, 3)}  Min Ret: {np.round(min_ret, 3)}')
            self.writer.add_scalar('Average return', avg_ret, iter)

    def update(self, sample_batch):
        data = sample_batch.to_tensor_dict()
        ret = data["ret"]
        val = data["val"]
        adv = ret - val
        data["adv"] = adv

        # Policy and Value loss
        self.policy.loss_gradient(data)
        self.policy.loss_value(data)


