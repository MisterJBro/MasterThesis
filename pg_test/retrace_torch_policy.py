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


def actor_critic_loss(
    policy: Policy,
    model: ModelV2,
    dist_class: ActionDistribution,
    train_batch: SampleBatch,
) -> TensorType:
    #print(train_batch["obs"].shape, train_batch[Postprocessing.ADVANTAGES].shape)
    if policy.is_recurrent():
        B = len(train_batch[SampleBatch.SEQ_LENS])
        max_seq_len = train_batch["obs"].shape[0] // B
        mask_orig = sequence_mask(train_batch[SampleBatch.SEQ_LENS], max_seq_len)
        valid_mask = torch.reshape(mask_orig, [-1])
    else:
        valid_mask = torch.ones(int(train_batch["obs"].shape[0]), dtype=torch.bool)

    # Q function loss
    if policy.config["use_qf"]:
        obs = train_batch["obs"].float()
        obs = obs.reshape(obs.shape[0], -1)

        trainset = torch.utils.data.TensorDataset(obs, torch.masked_select(train_batch[Postprocessing.VALUE_TARGETS], valid_mask), valid_mask, train_batch[SampleBatch.ACTIONS])
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
        obs = train_batch["obs"].float()
        obs = obs.reshape(obs.shape[0], -1)
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
                #print("\t", i, value_err)
    else:
        value_err = torch.zeros(1, requires_grad=True)

    # Policy loss
    logits, _ = model(train_batch)
    dist = dist_class(logits, model)
    log_probs = dist.logp(train_batch[SampleBatch.ACTIONS]).reshape(-1)
    pi_err = -torch.mean(
        torch.masked_select(
            log_probs * train_batch[Postprocessing.ADVANTAGES], valid_mask
        )
    )

    entropy = torch.mean(torch.masked_select(dist.entropy(), valid_mask))
    #print("Test: ", policy.config["vf_loss_coeff"])

    policy._optimizers[0].zero_grad()
    total_loss = pi_err# - entropy * policy.entropy_coeff #+ value_err * policy.config["vf_loss_coeff"]
    total_loss.backward()
    policy._optimizers[0].step()

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    model.tower_stats["entropy"] = entropy
    model.tower_stats["pi_err"] = pi_err
    model.tower_stats["value_err"] = value_err

    return (torch.zeros(1, requires_grad=True), torch.zeros(1, requires_grad=True), torch.zeros(1, requires_grad=True))


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
    loss_fn=actor_critic_loss,
    stats_fn=stats,
    make_model=build_model,
    postprocess_fn=compute_gae_for_sample_batch,
    extra_action_out_fn=vf_preds_fetches,
    extra_grad_process_fn=apply_grad_clipping,
    optimizer_fn=torch_optimizer,
    before_loss_init=setup_mixins,
    mixins=[ValueNetworkMixin, LearningRateSchedule, EntropyCoeffSchedule],
)