from src.train.config import create_config
from src.env.discretize_env import DiscreteActionWrapper
from src.env.pendulum import PendulumEnv
from src.train.pg import PGTrainer


if __name__ == '__main__':
    env = DiscreteActionWrapper(PendulumEnv())
    config = create_config({
        "env": env,
        "train_iters": 5,
        "num_cpus": 3,
        "num_envs": 30,
        "device": "cpu",
        "pi_lr": 8e-4,
        "vf_lr": 8e-4,
        "vf_iters": 5,
        "sample_len": 600,
    })

    with PGTrainer(config) as tr:
        tr.load()
        tr.train()
        tr.test()
        tr.save()