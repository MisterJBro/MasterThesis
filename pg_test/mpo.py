import logging
import random
from tkinter.tix import Tree
from typing import Any, Dict, Type

import numpy as np
import math
import ray
from scipy.optimize import minimize
from ray import tune
from ray.actor import ActorHandle
from ray.rllib.agents.trainer import Trainer, with_common_config
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.parallel_requests import asynchronous_parallel_requests
from ray.rllib.execution.rollout_ops import AsyncGradients
from ray.rllib.execution.train_ops import ApplyGradients
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.metrics import (
    APPLY_GRADS_TIMER,
    GRAD_WAIT_TIMER,
    NUM_AGENT_STEPS_SAMPLED,
    NUM_AGENT_STEPS_TRAINED,
    NUM_ENV_STEPS_SAMPLED,
    NUM_ENV_STEPS_TRAINED,
    SYNCH_WORKER_WEIGHTS_TIMER,
)
from ray.rllib.execution.common import (
    AGENT_STEPS_TRAINED_COUNTER,
    APPLY_GRADS_TIMER,
    COMPUTE_GRADS_TIMER,
    LAST_TARGET_UPDATE_TS,
    LEARN_ON_BATCH_TIMER,
    LOAD_BATCH_TIMER,
    NUM_TARGET_UPDATES,
    STEPS_SAMPLED_COUNTER,
    STEPS_TRAINED_COUNTER,
    STEPS_TRAINED_THIS_ITER_COUNTER,
    WORKER_UPDATE_TIMER,
    _check_sample_batch_type,
    _get_global_vars,
    _get_shared_metrics,
)
from ray.rllib.utils.metrics.learner_info import LearnerInfoBuilder
from ray.rllib.utils.typing import ResultDict, TrainerConfigDict
from ray.util.iter import LocalIterator
from ray.rllib.execution.rollout_ops import (
    ConcatBatches,
    ParallelRollouts,
    synchronous_parallel_sample,
)
from ray.rllib.execution.train_ops import (
    TrainOneStep,
    MultiGPUTrainOneStep,
    train_one_step,
    multi_gpu_train_one_step,
)
from ray.rllib.execution.common import WORKER_UPDATE_TIMER
from mpo_model import MPOModel

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = with_common_config({
    # Should use a critic as a baseline (otherwise don't use value baseline;
    # required for using GAE).
    "use_critic": True,
    # If true, use the Generalized Advantage Estimator (GAE)
    # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
    "use_gae": False,
    # Number of training iterations
    "iters": 100,
    # Size of rollout batch
    "rollout_fragment_length": 10,
    # Batch mode must be complete_episodes.
    #"batch_mode": "complete_episodes",
    # GAE(gamma) parameter
    "lambda": 0.97,
    # Max global norm for each gradient calculated by worker
    "grad_clip": 40.0,
    # Learning rates
    "lr": 0.0001,
    "vf_lr": 0.0001,
    "qf_lr": 0.0001,
    "vf_iters": 3,
    # Whether to use learn a Q function
    "use_qf": True,
    "qf_iters": 3,
    # Learning rate schedule
    "lr_schedule": None,
    # Entropy coefficient
    "entropy_coeff": 0.01,
    # Entropy coefficient schedule
    "entropy_coeff_schedule": None,
    "sample_async": False,
    # Sample batch size
    "sample_batch_size": 1_500,
    # Using retrace operator
    "retrace": True,

    # MPO
    # dual_constraint
    "ε_dual": 0.1,
    "ε_kl": 0.01,
    "α_max": 1.0,
    "α_scale": 10.0,
    "m_step_iters": 10,

    # Replay buffer
    "replay_buffer_config": {
         "_enable_replay_buffer_api": False,
        # How many steps of the model to sample before learning starts.
        #"learning_starts": 1_000,
        "type": "ray.rllib.utils.replay_buffers.replay_buffer.ReplayBuffer",
        # Capacity is counted in episodes (through some own changes)
        "capacity": 10_000,
        "storage_unit": "episodes",
    },

    # Custom Model
    "mpo_model": {
        "custom_model": MPOModel,
        # General Network Parameters
        "hidden_size": 256,
    },

    # Use the Trainer's `training_iteration` function instead of `execution_plan`.
    "_disable_execution_plan_api": True,
})


class MPOTrainer(Trainer):
    @classmethod
    @override(Trainer)
    def get_default_config(cls) -> TrainerConfigDict:
        return DEFAULT_CONFIG

    @override(Trainer)
    def validate_config(self, config: TrainerConfigDict) -> None:
        # Call super's validation method.
        super().validate_config(config)

        if config["entropy_coeff"] < 0:
            raise ValueError("`entropy_coeff` must be >= 0.0!")
        if config["num_gpus"] > 1:
            raise ValueError("No Multi GPU support for MPO!")

    @override(Trainer)
    def get_default_policy_class(self, config: TrainerConfigDict) -> Type[Policy]:
        if config["framework"] == "torch":
            from retrace_torch_policy import MPOTorchPolicy

            return MPOTorchPolicy
        else:
            raise ValueError("Unsupported framework: {}".format(config["framework"]))

    def sample_from_replay(self):
        sampled_size = 0
        samples = []
        while sampled_size < self.config["train_batch_size"]:
            i = random.randint(0, len(self.local_replay_buffer) - 1)
            self.local_replay_buffer._hit_count[i] += 1
            sample_episode = self.local_replay_buffer._storage[i]
            sampled_size += len(sample_episode)
            samples.append(sample_episode)

        sample_type = type(samples[0])
        out = sample_type.concat_samples(samples)
        out.decompress_if_needed()

        self.local_replay_buffer._num_timesteps_sampled += out.count
        return out

    def training_iteration(self) -> ResultDict:
        # Collect SampleBatches from sample workers until we have a full batch.
        if self._by_agent_steps:
            new_sample_batches = synchronous_parallel_sample(
                worker_set=self.workers, concat=False, max_agent_steps=self.config["sample_batch_size"]
            )
        else:
            new_sample_batches = synchronous_parallel_sample(
                worker_set=self.workers, concat=False, max_env_steps=self.config["sample_batch_size"]
            )

        # Add episodes of batches into replay buffer
        for batch in new_sample_batches:
            for eps in batch.split_by_episode():
                # Update sampling step counters.
                self._counters[NUM_ENV_STEPS_SAMPLED] += eps.env_steps()
                self._counters[NUM_AGENT_STEPS_SAMPLED] += eps.agent_steps()

                # Correct counter of replay buffer, internally they count each sample, but we want to count only each episode
                self.local_replay_buffer._num_timesteps_added -= eps.count - 1
                self.local_replay_buffer._num_timesteps_added_wrap -= eps.count - 1

                self.local_replay_buffer._add_single_batch(eps)

        # Sample one training MultiAgentBatch from replay buffer.
        train_batch = self.sample_from_replay()

        # Use simple optimizer (only for multi-agent or tf-eager; all other
        # cases should use the multi-GPU optimizer, even if only using 1 GPU).
        train_results = train_one_step(self, train_batch)

        # Update weights and global_vars - after learning on the local worker - on all
        # remote workers.
        global_vars = {
            "timestep": self._counters[NUM_ENV_STEPS_SAMPLED],
        }
        with self._timers[WORKER_UPDATE_TIMER]:
            self.workers.sync_weights(global_vars=global_vars)

        return train_results


if __name__ == "__main__":
    import os
    clear = lambda: os.system('cls')
    clear()

    # Configure the algorithm.
    config = {
        # === Settings for Rollout Worker processes ===
        "num_workers": 3,
        "num_envs_per_worker": 8,

        # === Settings for the Trainer process ===
        "iters": 300,
        "gamma": 0.99,
        "use_gae": False,
        "lambda": 0.9,

        "use_critic": False,
        "vf_lr": 2e-4,
        "vf_iters": 3,

        "use_qf": True,
        "qf_lr": 2e-4,
        "qf_iters": 5,

        "retrace": False,
        "lr": 2e-4,
        "entropy_coeff": 0.0,
        "rollout_fragment_length": 500,
        "sample_batch_size": 36_000,
        "train_batch_size": 36_000,

        # === Replay Buffer Settings ===
        "replay_buffer_config": {
            # Capacity is counted in episodes (through some own changes)
            "capacity": 10_000,
        },

        # === Environment Settings ===
        "env": "CartPole-v1",
        "env_config": {},

        # === Deep Learning Framework Settings ===
        "framework": "torch",
        "mpo_model": {
            "custom_model": MPOModel,
            # General Network Parameters
            "hidden_size": 256,
        },

        # === Evaluation Settings ===
        "evaluation_config": {
            "render_env": True,
        },
        "evaluation_num_workers": 1,

        # === Resource Settings ===
        "num_gpus": 0,
        "num_gpus_per_worker": 0,
        #"output": "logdir",
    }

    # Create our RLlib Trainer.
    trainer = MPOTrainer(config=config)

    # Train
    for i in range(config["iters"]):
        result = trainer.train()
        print(f"Iter: {i}, Reward: {result['episode_reward_mean']}")
    #tune.run(MPOTrainer, config=config, local_dir="./", stop={"episode_reward_mean": 500, "timesteps_total": 1_000_000}, checkpoint_at_end=True)

    # Evaluate
    trainer.evaluate()

    # During deletion of the trainers rollout workers some exceptions happen. Just ignore
    try:
        del trainer
    except Exception:
        pass
