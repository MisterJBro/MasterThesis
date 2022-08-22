from src.train.config import create_config
from src.train.trainer import Trainer

if __name__ == '__main__':
    config = create_config({
        "train_iters": 100,
        "env": "CartPole-v1",
        "num_cpus": 4,
        "num_envs": 20,
        "device": "cuda:0",
        "model_lr": 5e-4,
        "model_iters": 5,
        "model_unroll_len": 5,
        "grad_clip": 100.0,
    })

    with Trainer(config) as trainer:
        trainer.train()
        trainer.test()
