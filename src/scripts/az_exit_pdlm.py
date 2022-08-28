from src.train.config import create_config
from src.env.discretize_env import DiscreteActionWrapper
from src.env.pendulum import PendulumEnv
from src.train.exit import AZExitTrainer


if __name__ == '__main__':
    env = DiscreteActionWrapper(PendulumEnv())
    config = create_config({
        "env": env,
        "puct_c": 80.0,
        "train_iters": 50,
        "az_iters": 320,
        "az_eval_batch": 16,
        "num_cpus": 16,
        "num_envs": 16,
        "num_trees": 16,
        "device": "cuda:0",
        "pi_lr": 8e-4,
        "vf_lr": 8e-4,
        "sample_len": 500,
    })

    with AZExitTrainer(config) as tr:
        tr.train()
        tr.save()
        tr.test(render=False)
