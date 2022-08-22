from src.train.config import create_config
from src.env.discretize_env import DiscreteActionWrapper
from src.env.pendulum import PendulumEnv
from src.train.pg import PGTrainer


if __name__ == '__main__':
    env = DiscreteActionWrapper(PendulumEnv(), n_bins=11)
    config = create_config({
        "env": env,
        "train_iters": 150,
        "num_cpus": 3,
        "num_envs": 30,
        "device": "cpu",
        "pi_lr": 8e-4,
        "vf_lr": 5e-4,
        "sample_len": 500,
    })

    with PGTrainer(config) as tr:
        tr.train()
        tr.test()
        tr.save()