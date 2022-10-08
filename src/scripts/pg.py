from multiprocessing import freeze_support
from src.env.hex import HexEnv
from src.networks.residual import HexPolicy
from src.search.alpha_zero.alpha_zero import AlphaZero
from src.train.config import create_config
from src.train.pg import PGTrainer
from src.train.trainer import Trainer

if __name__ == '__main__':
    freeze_support()

    # Init for algos
    size = 9
    env = HexEnv(size)
    config = create_config({
        "train_iters": 100,
        "env": env,
        "puct_c": 20.0,
        "search_num_workers": 1,
        "search_evaluator_batch_size": 1,
        "search_return_adv": True,
        "num_cpus": 4,
        "num_envs": 16,
        "device": "cuda:0",
        "pi_lr": 1e-3,
        "vf_lr": 1e-3,
        "vf_iters": 2,
        "sample_len": 600,
    })

    # Import policy and model
    policy = HexPolicy(config)
    #policy.load("checkpoints/policy_pdlm_pgtrainer.pt")
    #model = ValueEquivalenceModel(config)
    #model.load("checkpoints/ve_model.pt")

    #az = AlphaZero(config, policy)

    with PGTrainer(config, policy) as trainer:
        trainer.train()
        trainer.test()
