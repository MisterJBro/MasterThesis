from src.networks.model import ValueEquivalenceModel
from src.networks.policy_pend import PendulumPolicy
from src.search.alpha_zero.alpha_zero import AlphaZero
from src.search.mcts.mcts import MCTS
from src.search.mu_zero.mu_zero import MuZero
from src.search.pgs.pgs import PGS
from src.search.ve_pgs.ve_pgs import VEPGS
from src.train.config import create_config
from src.env.discretize_env import DiscreteActionWrapper
from src.env.pendulum import PendulumEnv
from src.train.exit import ExitTrainer


if __name__ == '__main__':
    env = DiscreteActionWrapper(PendulumEnv())
    config = create_config({
        "env": env,
        "puct_c": 20.0,
        "train_iters": 100,
        "search_iters": 300,
        "search_num_workers": 15,
        "search_evaluator_batch_size": 15,
        "num_cpus": 15,
        "num_envs": 15,
        "device": "cuda:0",
        "pi_lr": 1e-3,
        "vf_lr": 5e-4,
        "sample_len": 500,
        "log_name": "log_az_exit_cuda.txt",
        "tree_output_qvals": True,
    })

    # Import policy and model
    policy = PendulumPolicy(config)
    model = ValueEquivalenceModel(config)

    # Algorithms
    az = AlphaZero(config, policy)
    #mz = MuZero(config, policy, model)
    #pgs = PGS(config, policy)
    #vepgs = VEPGS(config, policy, model)

    with ExitTrainer(config, az, policy) as tr:
        tr.train()
        tr.save()
        tr.test(render=False)
