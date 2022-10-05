import random
import torch
import numpy as np
from torch.multiprocessing import freeze_support
from src.networks.residual import HexPolicy
from src.search.alpha_zero.alpha_zero import AlphaZero
from src.train.config import create_config
from src.env.hex import HexEnv
from src.search.mcts.mcts import MCTS
from src.search.state import State


if __name__ == '__main__':
    freeze_support()

    # Init for algos
    size = 3
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
    policy = HexPolicy(config)
    #policy.load("checkpoints/policy_pdlm_pgtrainer.pt")
    #model = ValueEquivalenceModel(config)
    #model.load("checkpoints/ve_model.pt")

    # Algorithms /Players
    mcts = None
    def mcts_player(env, obs):
        global mcts
        if mcts is None:
            mcts = MCTS(config)
        dist = mcts.search(State(env, obs=obs), iters=200)
        act = env.available_actions()[np.argmax(dist)]
        return act

    az = None
    def az_player(env, obs):
        global az
        if az is None:
            az = AlphaZero(config, policy)
        dist = az.search(State(env, obs=obs), iters=200)
        act = env.available_actions()[np.argmax(dist)]
        return act

    def human_player(env, obs):
        available_actions = env.available_actions()
        while True:
            act = int(input("Please type in your action: "))
            if act in [np.argmax(a) for a in available_actions]:
                act = np.eye(size*size)[act]
                break
        return act

    def random_player(env, obs):
        return random.choice(env.available_actions())

    def nn_player(env, obs):
        # obs (2, 9, 9, 1)
        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist, val = policy(obs, legal_actions=[env.available_actions(use_indices=True)])
        act = dist.logits.argmax(-1).cpu().numpy()[0]
        act = np.eye(size*size)[act]
        return act

    # Simulate
    players = [nn_player, random_player]
    obs = env.reset()

    done = False
    black_turn = True
    while not done:
        env.render()

        if black_turn:
            print('Player 0')
            act = players[0](env, obs)
        else:
            print('Player 1')
            act = players[1](env, obs)

        obs, reward, done, _ = env.step(act)

        black_turn = not black_turn

    env.render()
    black_turn = not black_turn
    print(f"Winner is {'Player 0' if black_turn else f'Player 1'}!")

    # Close
    if mcts is not None:
        mcts.close()
    if az is not None:
        az.close()
