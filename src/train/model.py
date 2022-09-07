import numpy as np
from src.networks.policy_pend import PendulumPolicy
from src.train.trainer import Trainer
from src.networks.model import ValueEquivalenceModel
from src.search.model_search import plan


class ModelTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)

        self.model = ValueEquivalenceModel(config)

    def update(self, sample_batch):
        data = sample_batch.to_tensor_dict()
        obs = data["obs"]
        ret = data["ret"]
        val = data["val"]

        # Time
        start = time.time()
        plan_targets = plan(self.policy, self.model, data, self.config)
        plan_actions = plan_targets.logits.argmax(-1)
        end = time.time()
        #print(f'Plan time: {end - start}')

        # Distill planning targets into policy
        trainset = TensorDataset(obs, plan_targets.logits)
        trainloader = DataLoader(trainset, batch_size=int(self.config["num_samples"]/10), shuffle=True)

        for i in range(20):
            for obs_batch, plan_target_batch in trainloader:
                self.policy.opt_policy.zero_grad()
                dist_batch = self.policy.get_dist(obs_batch)
                loss = kl_divergence(Categorical(logits=plan_target_batch), dist_batch).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.policy.parameters(),  self.config["grad_clip"])
                self.policy.opt_policy.step()

        #act = data["act"]
        #act_onehot = to_onehot(act, self.config["num_acts"])
        #with torch.no_grad():
        #    act_model = self.model.dyn_linear(act_onehot)
        #    state = self.model.representation(obs)
        #    hidden, _ = self.model.dynamics(state, act_model)
        #    model_val = self.model.get_value(hidden)
        #    model_rew = self.model.get_reward(hidden)
        #q_val = model_rew + self.config["gamma"] * model_val

        adv = ret - val
        data["adv"] = adv

        # Policy and Value loss
        #self.policy.loss_gradient(data)
        self.policy.loss_value(data)

        # Get new logits for model loss
        with torch.no_grad():
            data["logits"] = self.policy.get_dist(obs).logits

        # Model loss
        self.model.loss(data)



