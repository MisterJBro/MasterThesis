from typing import Dict, List, Type, Union, Optional, Tuple

import ray
from ray.rllib.agents.trainer import Trainer
from ray.rllib.agents.trainer_config import TrainerConfig
from ray.rllib.agents.pg.utils import post_process_advantages
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.typing import TrainerConfigDict, TensorType
from ray.rllib.agents.pg import PGTrainer

torch, nn = try_import_torch()

def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
    """Returns the calculated loss in a stats dict.
    Args:
        policy: The Policy object.
        train_batch: The data used for training.
    Returns:
        Dict[str, TensorType]: The stats dict.
    """

    return convert_to_numpy({
        "policy_loss": torch.mean(
            torch.stack(self.get_tower_stats("policy_loss"))
        ),
    })

class PGTestTorchPolicy(TorchPolicy):
    """PyTorch policy class used with PGTrainer."""

    def __init__(self, observation_space, action_space, config):
        config = dict(PGTestConfig().to_dict(), **config)

        TorchPolicy.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"],
        )

        # TODO: Don't require users to call this manually.
        self._initialize_loss_from_dummy_batch(
            auto_remove_unneeded_view_reqs=True,
            stats_fn=stats_fn,
        )

    @override(TorchPolicy)
    def loss(
        self,
        model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        """The basic policy gradients loss function.
        Calculates the vanilla policy gradient loss based on:
        L = -E[ log(pi(a|s)) * A]
        Args:
            model: The Model to calculate the loss for.
            dist_class: The action distr. class.
            train_batch: The training data.
        Returns:
            Union[TensorType, List[TensorType]]: A single loss tensor or a list
                of loss tensors.
        """
        # Pass the training data through our model to get distribution parameters.
        dist_inputs, _ = model(train_batch)

        # Create an action distribution object.
        action_dist = dist_class(dist_inputs, model)

        # Calculate the vanilla PG loss based on:
        # L = -E[ log(pi(a|s)) * A]
        log_probs = action_dist.logp(train_batch[SampleBatch.ACTIONS])

        # Final policy loss.
        policy_loss = -torch.mean(log_probs * train_batch[Postprocessing.ADVANTAGES])

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["policy_loss"] = policy_loss

        return policy_loss

    @override(TorchPolicy)
    def postprocess_trajectory(
        self,
        sample_batch: SampleBatch,
        other_agent_batches = None,
        episode: Optional["Episode"] = None,
    ) -> SampleBatch:
        sample_batch = super().postprocess_trajectory(
            sample_batch, other_agent_batches, episode
        )
        return post_process_advantages(self, sample_batch, other_agent_batches, episode)

class PGTestConfig(TrainerConfig):
    """Configuration variables for Trainer"""
    def __init__(self):
        super().__init__(trainer_class=PGTest)

        # Override some of AlgorithmConfig's default values
        self.num_workers = 0
        self.lr = 0.0004
        self._disable_preprocessor_api = True
        self.framework = "torch"

class PGTest(Trainer):
    """Policy Gradient (PG) Trainer"""

    @override(Trainer)
    def setup(self, config):
        # Initialization of algorithm
        super().setup(config)

    @classmethod
    @override(Trainer)
    def get_default_config(cls) -> TrainerConfigDict:
        # Default configuration of algorithm (if no config is given)
        return PGTestConfig().to_dict()

    @override(Trainer)
    def validate_config(self, config: TrainerConfigDict) -> None:
        # Check if given config is valid e.g. no negative learning rate
        super().validate_config(config)

    @override(Trainer)
    def get_default_policy_class(self, config) -> Type[Policy]:
        if config["framework"] == "torch":
            return PGTestTorchPolicy
        else:
            raise ValueError("Unsupported framework: {}".format(config["framework"]))

if __name__ == "__main__":
    # Configure the algorithm.
    config = {
        # === Settings for Rollout Worker processes ===
        "num_workers": 3,
        "num_envs_per_worker": 32,

        # === Settings for the Trainer process ===
        #"gamma": 0.99,
        #"lr": 0.0001,
        "train_batch_size": 3*32*800,
        #"model": {
        #    "fcnet_hiddens": [256, 256],
        #    "fcnet_activation": "tanh",
        #    #"post_fcnet_hiddens": [128],
        #    #"post_fcnet_activation": "tanh",
        #    "vf_share_layers": False,
        #},

        # === Environment Settings ===
        "env": "CartPole-v1",
        "env_config": {},

        # === Deep Learning Framework Settings ===
        "framework": "torch",

        # === Evaluation Settings ===
        "evaluation_config": {
            "render_env": True,
        },
        "evaluation_num_workers": 1,

        # === Resource Settings ===
        "num_gpus":1,
        "num_gpus_per_worker": 0,
    }

    # Create our RLlib Trainer.
    PGTestTrainer = PGTest
    trainer = PGTestTrainer(config=config)

    # Run it for n training iterations. A training iteration includes
    # parallel sample collection by the environment workers as well as
    # loss calculation on the collected batch and a model update.
    for _ in range(200):
        result = trainer.train()
        print(result["episode_reward_mean"])

    # Evaluate the trained Trainer (and render each timestep to the shell's
    # output).
    print('EVAL')
    trainer.evaluate()

    print('END')
