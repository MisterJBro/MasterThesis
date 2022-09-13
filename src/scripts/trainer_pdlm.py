from src.train.config import create_config
from src.env.discretize_env import DiscreteActionWrapper
from src.env.pendulum import PendulumEnv
from src.train.pg import PGTrainer
from src.train.model import ModelTrainer


if __name__ == '__main__':
    env = DiscreteActionWrapper(PendulumEnv())
    config = create_config({
        "env": env,
        "train_iters": 40,
        "num_cpus": 3,
        "num_envs": 30,
        "device": "cpu",
        "pi_lr": 1e-3,
        "vf_lr": 1e-3,
        "vf_iters": 2,
        "sample_len": 600,
    })

    with ModelTrainer(config) as tr:
        #tr.load("checkpoints/policy_pdlm_pgtrainer.pt")
        tr.train()
        tr.test(render=False)
        tr.save()