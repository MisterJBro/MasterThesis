from src.train.config import create_config
from src.env.discretize_env import DiscreteActionWrapper
from src.env.pendulum import PendulumEnv
from src.train.exit import AZExitTrainer


if __name__ == '__main__':
    env = DiscreteActionWrapper(PendulumEnv())
    config = create_config({
        "env": env,
        "puct_c": 5.0,
        "train_iters": 50,
        "az_iters": 300,
        "az_eval_batch": 15,
        "num_cpus": 3,
        "num_envs": 15,
        "num_trees": 15,
        "device": "cpu",
        "pi_lr": 8e-4,
        "vf_lr": 8e-4,
        "sample_len": 500,
    })

    with AZExitTrainer(config) as tr:
        tr.train()
        tr.save()
        tr.test()
