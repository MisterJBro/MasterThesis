from cmath import log
import gym
import numpy as np
from typing import Dict, List, Optional

import ray
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.postprocessing import (
    compute_gae_for_sample_batch,
    Postprocessing,
)
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_template import build_policy_class
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import LearningRateSchedule, EntropyCoeffSchedule
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import apply_grad_clipping, sequence_mask
from ray.rllib.utils.typing import (
    TrainerConfigDict,
    TensorType,
    PolicyID,
    LocalOptimizer,
)
from ray.rllib.models.catalog import ModelCatalog

torch, nn = try_import_torch()
from torch.nn.functional import mse_loss
from scipy.optimize import minimize

# calculates KL between two Categorical distributions from https://github.com/daisatojp/mpo/blob/master/mpo/mpo.py
def categorical_kl(p1, p2):
    p1 = torch.clamp_min(p1, 0.0001)
    p2 = torch.clamp_min(p2, 0.0001)
    return torch.mean((p1 * torch.log(p1 / p2)).sum(dim=-1))

def retrace(
    policy: Policy,
    model: ModelV2,
    dist_class: ActionDistribution,
    train_batch: SampleBatch,
):
    obs = train_batch["obs"].float()
    obs = obs.reshape(obs.shape[0], -1)

    with torch.no_grad():
        logits, _ = model(train_batch)
        dist = dist_class(logits, model)
        probs = dist.dist.probs
        log_probs = dist.logp(train_batch[SampleBatch.ACTIONS]).reshape(-1)
    old_log_probs = train_batch['action_logp']

    with torch.no_grad():
        model.obs_flat = obs
        q_values = model.q_function()
        v_values = (probs * q_values).sum(1, keepdim=True)
        q_single = q_values.gather(1, train_batch[SampleBatch.ACTIONS].unsqueeze(1)).squeeze(1)
    train_batch['q_values'] = q_values
    train_batch['v_values'] = v_values
    train_batch['q_single'] = q_single

    log_rho = log_probs - old_log_probs
    log_c = torch.min(torch.zeros_like(log_rho), log_rho)
    trace = torch.exp(log_c)
    train_batch['trace'] = trace

    Q_TARGETS = []
    gamma = policy.config['gamma']
    # Iterate over all episodes and calculate retrace operator
    for eps in train_batch.split_by_episode():
        c_s = eps['trace']
        r_t = torch.tensor(eps['rewards'])
        Eq = torch.concat([eps['v_values'].reshape(-1)[1:], torch.tensor([0])])
        q_t = eps['q_single']
        if eps.get(SampleBatch.DONES)[-1]:
            eps['q_single'][-1] = 0
        err = r_t + gamma * Eq - q_t

        #Calculate retrace targets
        local_targets = []
        q_ret = torch.zeros(1)
        for i in reversed(range(len(eps))):
            q_sum = err[i] + q_ret
            local_target = q_t[i] + q_sum
            local_targets.append(local_target)
            q_ret = gamma * c_s[i] * q_sum
        local_targets = list(reversed(local_targets))
        Q_TARGETS.extend(local_targets)
    Q_TARGETS = torch.tensor(Q_TARGETS)
    return Q_TARGETS

def mpo_loss(
    policy: Policy,
    model: ModelV2,
    dist_class: ActionDistribution,
    train_batch: SampleBatch,
) ->  TensorType:

    # Reshape observations
    obs = train_batch["obs"].float()
    obs = obs.reshape(obs.shape[0], -1)
    valid_mask = torch.ones(int(train_batch["obs"].shape[0]), dtype=torch.bool)

    if policy.config["retrace"]:
        Q_TARGETS = retrace(policy, model, dist_class, train_batch)
    else:
        Q_TARGETS = train_batch[Postprocessing.VALUE_TARGETS]

    #if policy.is_recurrent():
    #    B = len(train_batch[SampleBatch.SEQ_LENS])
    #    max_seq_len = train_batch["obs"].shape[0] // B
    #    mask_orig = sequence_mask(train_batch[SampleBatch.SEQ_LENS], max_seq_len)
    #    valid_mask = torch.reshape(mask_orig, [-1])
    #else:
    #

    # Q function loss
    if policy.config["use_qf"]:
        trainset = torch.utils.data.TensorDataset(obs, torch.masked_select(Q_TARGETS, valid_mask), valid_mask, train_batch[SampleBatch.ACTIONS])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=int(policy.config["train_batch_size"]/10), shuffle=True)
        for i in range(policy.config["qf_iters"]):
            for obs_flat, qf_target, batch_valid_mask, actions in trainloader:
                policy._optimizers[2].zero_grad()
                model.obs_flat = obs_flat
                q_values = model.q_function()
                q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                q_values = torch.masked_select(q_values, batch_valid_mask)

                q_value_err = mse_loss(q_values, qf_target)
                q_value_err.backward()
                policy._optimizers[2].step()
                #print("\t", i, q_value_err)

        model.obs_flat = obs
        q_values = model.q_function()
        q_values = q_values.gather(1, train_batch[SampleBatch.ACTIONS].unsqueeze(1)).squeeze(1)
        q_values = torch.masked_select(q_values, valid_mask)
        train_batch[Postprocessing.ADVANTAGES] = q_values

    # Value function loss
    if policy.config["use_critic"]:
        trainset = torch.utils.data.TensorDataset(obs, torch.masked_select(train_batch[Postprocessing.VALUE_TARGETS], valid_mask), valid_mask)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=int(policy.config["train_batch_size"]/10), shuffle=True)

        for i in range(policy.config["vf_iters"]):
            for obs_flat, vf_target, batch_valid_mask in trainloader:
                policy._optimizers[1].zero_grad()
                model.obs_flat = obs_flat
                values = model.value_function()
                values = torch.masked_select(values.reshape(-1), batch_valid_mask)

                value_err = mse_loss(values, vf_target)
                value_err.backward()
                policy._optimizers[1].step()
    else:
        value_err = torch.zeros(1, requires_grad=True)

    trainset = torch.utils.data.TensorDataset(obs)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=int(policy.config["train_batch_size"]/100), shuffle=True)
    #print(trainloader.next())

    # E-Step adapted from https://github.com/daisatojp/mpo/blob/master/mpo/mpo.py
    da = model.action_num
    K = train_batch["obs"].shape[0]
    actions = torch.arange(da)[..., None].expand(model.action_num, K)
    with torch.no_grad():
        logits, _ = model({"obs": obs})
        dist = dist_class(logits, model)
        b_prob = dist.dist.expand((da, K)).log_prob(actions).exp()
        b_prob_np = b_prob.cpu().transpose(0, 1).numpy()

        q_values = model.q_function().transpose(0, 1)
        target_q_np = q_values.cpu().transpose(0, 1).numpy()


    # Dual function
    def dual(η):
        max_q = np.max(target_q_np, 1)
        return η * policy.config["ε_dual"] + np.mean(max_q) \
            + η * np.mean(np.log(np.sum(
                b_prob_np * np.exp((target_q_np - max_q[:, None]) / η), axis=1)))

    bounds = [(1e-6, None)]
    res = minimize(dual, np.array([model.η]), method='SLSQP', bounds=bounds)
    model.η = res.x[0]
    qij = torch.softmax(q_values / model.η, dim=0)

    """
    # Policy loss
    logits, _ = model(train_batch)
    dist = dist_class(logits, model)
    log_probs = dist.logp(train_batch[SampleBatch.ACTIONS]).reshape(-1)
    pi_err = -torch.mean(
        torch.masked_select(
            log_probs * train_batch[Postprocessing.ADVANTAGES], valid_mask
        )
    )
    policy._optimizers[0].zero_grad()
    total_loss = pi_err
    total_loss.backward()
    policy._optimizers[0].step()

    model.tower_stats["entropy"] = torch.zeros(1, requires_grad=True)
    model.tower_stats["pi_err"] = torch.zeros(1, requires_grad=True)
    model.tower_stats["value_err"] = torch.zeros(1, requires_grad=True)

    return (model.tower_stats["entropy"], model.tower_stats["pi_err"], model.tower_stats["value_err"])
    """

    # M-Step
    mean_loss_p = []
    mean_loss_l = []
    max_kl = []
    for _ in range(policy.config["m_step_iters"]):
        logits, _ = model({"obs": obs})
        dist = dist_class(logits, model)
        log_probs = dist.logp(actions)
        prob = torch.exp(dist.logp(train_batch[SampleBatch.ACTIONS]))
        loss_p = torch.mean(qij * log_probs)
        mean_loss_p.append((-loss_p).item())

        kl = categorical_kl(p1=prob, p2=b_prob)
        max_kl.append(kl.item())

        if np.isnan(kl.item()):  # This should not happen
            raise RuntimeError('kl is nan')

        model.α -= policy.config["α_scale"] * (policy.config["ε_kl"] - kl).detach().item()
        model.α = np.clip(model.α, 0.0, policy.config["α_max"])

        policy._optimizers[0].zero_grad()
        # last eq of [2] p.5
        loss_l = -(loss_p + model.α * (policy.config["ε_kl"] - kl))
        mean_loss_l.append(loss_l.item())
        loss_l.backward()
        policy._optimizers[0].step()

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["entropy"] = torch.zeros(1, requires_grad=True)
    model.tower_stats["pi_err"] = torch.zeros(1, requires_grad=True)
    model.tower_stats["value_err"] = torch.zeros(1, requires_grad=True)

    return (model.tower_stats["entropy"], model.tower_stats["pi_err"], model.tower_stats["value_err"])


def stats(policy: Policy, train_batch: SampleBatch) -> Dict[str, TensorType]:

    return {
        "cur_lr": policy.cur_lr,
        "entropy_coeff": policy.entropy_coeff,
        "policy_entropy": torch.mean(torch.stack(policy.get_tower_stats("entropy"))),
        "policy_loss": torch.mean(torch.stack(policy.get_tower_stats("pi_err"))),
        "vf_loss": torch.mean(torch.stack(policy.get_tower_stats("value_err"))),
    }


def vf_preds_fetches(
    policy: Policy,
    input_dict: Dict[str, TensorType],
    state_batches: List[TensorType],
    model: ModelV2,
    action_dist: TorchDistributionWrapper,
) -> Dict[str, TensorType]:
    """Defines extra fetches per action computation.
    Args:
        policy (Policy): The Policy to perform the extra action fetch on.
        input_dict (Dict[str, TensorType]): The input dict used for the action
            computing forward pass.
        state_batches (List[TensorType]): List of state tensors (empty for
            non-RNNs).
        model (ModelV2): The Model object of the Policy.
        action_dist (TorchDistributionWrapper): The instantiated distribution
            object, resulting from the model's outputs and the given
            distribution class.
    Returns:
        Dict[str, TensorType]: Dict with extra tf fetches to perform per
            action computation.
    """
    # Return value function outputs. VF estimates will hence be added to the
    # SampleBatches produced by the sampler(s) to generate the train batches
    # going into the loss function.
    return {
        SampleBatch.VF_PREDS: model.value_function(),
    }


def torch_optimizer(policy: Policy, config: TrainerConfigDict) -> LocalOptimizer:
    actor_weights = policy.model.actor.parameters()
    value_weights = policy.model.value.parameters()
    q_weights = policy.model.q.parameters()

    actor_opt = torch.optim.Adam(actor_weights, lr=config["lr"])
    value_opt = torch.optim.Adam(value_weights, lr=config["vf_lr"])
    q_opt = torch.optim.Adam(q_weights, lr=config["qf_lr"])

    return (actor_opt, value_opt, q_opt)

def build_model(policy, obs_space, action_space, config):
    model = ModelCatalog.get_model_v2(
        obs_space,
        action_space,
        action_space.n,
        config["mpo_model"],
        name="MPOModel",
        framework="torch",
    )
    policy.model_variables = model.variables()

    return model



class ValueNetworkMixin:
    """Assigns the `_value()` method to the PPOPolicy.
    This way, Policy can call `_value()` to get the current VF estimate on a
    single(!) observation (as done in `postprocess_trajectory_fn`).
    Note: When doing this, an actual forward pass is being performed.
    This is different from only calling `model.value_function()`, where
    the result of the most recent forward pass is being used to return an
    already calculated tensor.
    """

    def __init__(self, obs_space, action_space, config):
        # When doing GAE, we need the value function estimate on the
        # observation.
        if config["use_gae"]:
            # Input dict is provided to us automatically via the Model's
            # requirements. It's a single-timestep (last one in trajectory)
            # input_dict.

            def value(**input_dict):
                input_dict = SampleBatch(input_dict)
                input_dict = self._lazy_tensor_dict(input_dict)
                model_out, _ = self.model(input_dict)
                # [0] = remove the batch dim.
                return self.model.value_function()[0].item()

        # When not doing GAE, we do not require the value function's output.
        else:

            def value(*args, **kwargs):
                return 0.0

        self._value = value


def setup_mixins(
    policy: Policy,
    obs_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    config: TrainerConfigDict,
) -> None:
    """Call all mixin classes' constructors before PPOPolicy initialization.
    Args:
        policy (Policy): The Policy object.
        obs_space (gym.spaces.Space): The Policy's observation space.
        action_space (gym.spaces.Space): The Policy's action space.
        config (TrainerConfigDict): The Policy's config.
    """
    EntropyCoeffSchedule.__init__(
        policy, config["entropy_coeff"], config["entropy_coeff_schedule"]
    )
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)


from mpo import DEFAULT_CONFIG as MPO_DEFAULT_CONFIG

MPOTorchPolicy = build_policy_class(
    name="MPOTorchPolicy",
    framework="torch",
    get_default_config=lambda: MPO_DEFAULT_CONFIG,
    apply_gradients_fn=lambda a, b: None,
    loss_fn=mpo_loss,
    stats_fn=stats,
    make_model=build_model,
    postprocess_fn=compute_gae_for_sample_batch,
    extra_action_out_fn=vf_preds_fetches,
    extra_grad_process_fn=apply_grad_clipping,
    optimizer_fn=torch_optimizer,
    before_loss_init=setup_mixins,
    mixins=[ValueNetworkMixin, LearningRateSchedule, EntropyCoeffSchedule],
)