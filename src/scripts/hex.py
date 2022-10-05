import random
import numpy as np
from torch.multiprocessing import freeze_support
from src.search.alpha_zero.alpha_zero import AlphaZero
from src.train.config import create_config
from src.env.hex import HexEnv
from src.search.mcts.mcts import MCTS
from src.search.state import State


if __name__ == '__main__':
    freeze_support()

    # Init for algos
    size = 9
    env = HexEnv(size)
    config = create_config({
        "env": env,
        "puct_c": 20.0,
        "uct_c": 20.0,
        "search_num_workers": 1,
        "search_evaluator_batch_size": 1,
        "dirichlet_eps": 0.0,
        "pgs_lr": 1e-1,
        "pgs_trunc_len": 5,
        "device": "cpu",
        "search_return_adv": True,
    })


    # Import policy and model
    policy = PendulumPolicy(config)
    policy.load("checkpoints/policy_pdlm_pgtrainer.pt")
    #model = ValueEquivalenceModel(config)
    #model.load("checkpoints/ve_model.pt")

    # Algorithms /Players
    mcts = MCTS(config)
    def mcts_player(env, obs):
        dist = mcts.search(State(env, obs=obs), iters=200)
        act = env.available_actions()[np.argmax(dist)]
        return act

    az = AlphaZero(config, policy)
    def az_player(env, obs):
        dist = az.search(State(env, obs=obs), iters=200)
        act = env.available_actions()[np.argmax(dist)]
        return act

    def human_player(env, obs):
        while True:
            act = int(input("Please type in your action: "))
            if act in [np.argmax(a) for a in env.available_actions()]:
                act = np.eye(size*size)[act]
                break
        return act

    def random_player(env, obs):
        return random.choice(env.available_actions())

    # Simulate
    obs = env.reset()

    done = False
    black_turn = True
    while not done:
        env.render()

        if black_turn:
            act = random_player(env, obs)
        else:
            act = mcts_player(env, obs)

        obs, reward, done, _ = env.step(act)

        black_turn = not black_turn

    env.render()
    black_turn = not black_turn
    print(f"Winner is {'random' if black_turn else 'mcts'}!")

    # Close
    mcts.close()
