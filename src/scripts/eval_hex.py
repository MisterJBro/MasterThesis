import random
import torch
import numpy as np
from torch.multiprocessing import freeze_support
from src.networks.residual import HexPolicy
from src.search.alpha_zero.alpha_zero import AlphaZero
from src.search.pgs.pgs import PGS
from src.train.config import create_config
from src.env.hex import HexEnv
from src.search.mcts.mcts import MCTS
from src.search.state import State
from tqdm import tqdm, trange


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
    policy = HexPolicy(config)
    #policy.load("checkpoints/policy_pdlm_pgtrainer.pt")
    #model = ValueEquivalenceModel(config)
    #model.load("checkpoints/ve_model.pt")

    # Algorithms /Players
    mcts_obj = None
    def mcts(env, obs):
        global mcts_obj
        if mcts_obj is None:
            mcts_obj = MCTS(config)
        result = mcts_obj.search(State(env, obs=obs), iters=200)
        act = env.available_actions()[np.argmax(result)]
        return act

    az_obj = None
    def az(env, obs):
        global az_obj
        if az_obj is None:
            az_obj = AlphaZero(config, policy)
        result = az_obj.search(State(env, obs=obs), iters=200)
        act = env.available_actions()[np.argmax(result)]
        return act

    pgs_obj = None
    def pgs(env, obs):
        global pgs_obj
        if pgs_obj is None:
            pgs_obj = PGS(config, policy)
        result = pgs_obj.search(State(env, obs=obs), iters=200)
        act = env.available_actions()[np.argmax(result)]
        return act

    def human(env, obs):
        available_actions = env.available_actions()
        while True:
            act = int(input("Please type in your action: "))
            if act in [np.argmax(a) for a in available_actions]:
                break
        return act

    def random(env, obs):
        return random.choice(env.available_actions())

    def nn(env, obs):
        # obs (2, 9, 9, 1)
        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist, val = policy(obs, legal_actions=[env.available_actions()])
        act = dist.logits.argmax(-1).cpu().numpy()[0]
        return act

    # Simulate
    players = [az, nn]
    num_games = 1
    render = False
    num_victories_first = 0
    print(f"Simulating games: {players[0].__name__.upper()} vs {players[1].__name__.upper()}!")
    for i in trange(num_games):
        obs = env.reset()

        done = False
        black_turn = True
        while not done:
            if render:
                env.render()

            if black_turn:
                act = players[0](env, obs)
            else:
                act = players[1](env, obs)

            obs, reward, done, _ = env.step(act)

            black_turn = not black_turn

        if render:
            env.render()
        black_turn = not black_turn
        if black_turn:
            num_victories_first += 1


    print(f"Winrate {players[0].__name__.upper()}: {num_victories_first / num_games :.02f}")

    # Close
    if mcts_obj is not None:
        mcts_obj.close()
    if az_obj is not None:
        az_obj.close()