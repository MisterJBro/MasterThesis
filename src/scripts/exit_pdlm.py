from src.networks.model import ValueEquivalenceModel
from src.networks.residual import PendulumPolicy
from src.search.alpha_zero.alpha_zero import AlphaZero
from src.search.mcts.mcts import MCTS
from src.search.mu_zero.mu_zero import MuZero
from src.search.pgs.pgs import PGS
from src.search.ve_pgs.ve_pgs import VEPGS
from src.train.config import create_config
from src.env.discretize_env import DiscreteActionWrapper
from src.env.pendulum import PendulumEnv
from src.train.exit import ExitTrainer
import argparse


parser = argparse.ArgumentParser(description='File to train a model using Expert Iteration')

# Parser arguments
parser.add_argument('--search_algo', type=str, default="az", choices=['az', 'mz', 'pgs', 'vepgs'])

if __name__ == '__main__':
    args = parser.parse_args()
    search_algo = args.search_algo

    env = DiscreteActionWrapper(PendulumEnv())
    config = create_config({
        "env": env,
        "puct_c": 80.0,
        "train_iters": 50,
        "search_iters": 300,
        "search_num_workers": 15,
        "search_evaluator_batch_size": 15,
        "num_workers": 15,
        "num_envs": 15,
        "device": "cpu",
        "pi_lr": 1e-3,
        "vf_lr": 1e-3,
        "sample_len": 500,
        "search_return_adv": True,
        "log_name": f"{search_algo}_exit_log.txt",
        "log_to_file": True,
    })

    # Import policy and model
    policy = PendulumPolicy(config)
    model = None

    # Algorithms
    if search_algo == "az":
        search_algo = AlphaZero(config, policy)
    elif search_algo == "mz":
        model = ValueEquivalenceModel(config)
        search_algo = MuZero(config, policy, model)
    elif search_algo == "pgs":
        search_algo = PGS(config, policy)
    elif search_algo == "vepgs":
        model = ValueEquivalenceModel(config)
        search_algo = VEPGS(config, policy, model)
    else:
        raise ValueError('Unknown search algorithm: ', search_algo)

    # Train
    with ExitTrainer(config, search_algo, policy, model=model) as tr:
        tr.train()
        tr.save()
        tr.test(render=False)
