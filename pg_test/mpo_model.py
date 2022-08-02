import numpy as np
from typing import Any, List, Tuple, Dict
from ray.rllib.models.torch.misc import Reshape
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.framework import TensorType
from ray.rllib.utils.annotations import override

torch, nn = try_import_torch()

# Custom Model
class MPOModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.obs_size = int(np.product(obs_space.shape))
        self.hidden_size = model_config["hidden_size"]

        # Lagrange Multiplier
        self.η = np.random.rand()
        self.α = 0.0
        self .action_num = num_outputs

        self.actor = nn.Sequential(
            nn.Linear(self.obs_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, num_outputs)
        )
        self.value = nn.Sequential(
            nn.Linear(self.obs_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )
        self.q = nn.Sequential(
            nn.Linear(self.obs_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, num_outputs)
        )

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        #print(self.device)
        self.obs_flat = None

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType,) -> Tuple[TensorType, List[float], List[TensorType]]:
        obs = input_dict["obs"].float()
        self.obs_flat = obs.reshape(obs.shape[0], -1)
        logits = self.actor(self.obs_flat)
        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        value = self.value(self.obs_flat)
        return value.squeeze(1)

    def q_function(self) -> TensorType:
        q_value = self.q(self.obs_flat)
        return q_value.squeeze(1)