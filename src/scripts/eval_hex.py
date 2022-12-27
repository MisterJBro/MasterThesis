import random as random_lib
from src.networks.residual_model import ValueEquivalenceModel
import torch
import numpy as np
from torch.multiprocessing import freeze_support
from src.networks.residual import HexPolicy
from src.search.alpha_zero.alpha_zero import AlphaZero
from src.search.ve_pgs.ve_pgs import VEPGS
from src.search.mu_zero.mu_zero import MuZero
from src.search.pgs.pgs import PGS
from src.train.config import create_config
from src.env.hex import HexEnv
from src.search.mcts.mcts import MCTS
from src.search.state import State
from tqdm import tqdm, trange
from copy import deepcopy
from hexgame import RustMCTS, RustState


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    freeze_support()

    # Init for algos
    size = 7
    env = HexEnv(size)
    config = create_config({
        "env": env,
        "puct_c": 3.0,
        "uct_c": np.sqrt(2),
        "search_num_workers": 1,
        "search_evaluator_batch_size": 1,
        "dirichlet_eps": 0.0,
        "pgs_lr": 1e-1,
        "pgs_trunc_len": 5,
        "device": "cpu",
        "search_return_adv": True,

        "num_res_blocks": 16,
        "num_filters": 128,
        "model_num_res_blocks": 10,
        "model_num_filters": 128,
    })

    # Import policy and model
    policy1 = HexPolicy(config)
    policy1.load("hall_of_fame/p_7x7_16_128.pt")
    policy1.eval()
    policy2 = HexPolicy(config)
    #policy2.load("checkpoints/policy_hex_9x9_2.pt")
    policy2.eval()
    model = ValueEquivalenceModel(config)
    #model.load("checkpoints/m_5x5_10_128.pt")

    # Algorithms /Players
    mcts_obj = None
    def mcts(env, obs, info):
        global mcts_obj
        if mcts_obj is None:
            mcts_obj = MCTS(config)
        result = mcts_obj.search(State(env, obs=obs), iters=1_000)
        act = np.argmax(result["pi"])
        return act

    mcts_rust_obj = None
    def mcts_rust(env, obs, info):
        global mcts_rust_obj
        if mcts_rust_obj is None:
            mcts_rust_obj = RustMCTS(1)
        state_rust = RustState(env.env, obs, info["pid"], info["legal_act"])
        result = mcts_rust_obj.search(state_rust, iters=1_000)
        act = np.argmax(result["pi"])
        return act

    az_obj = None
    def az(env, obs, info):
        global az_obj
        if az_obj is None:
            az_obj = AlphaZero(config, policy1)
        result = az_obj.search(State(env, obs=obs), iters=10_000)
        print(result["V"])
        act = np.argmax(result["pi"])
        return act

    pgs_obj = None
    def pgs(env, obs, info):
        global pgs_obj
        if pgs_obj is None:
            pgs_obj = PGS(config, policy1)
        result = pgs_obj.search(State(env, obs=obs), iters=100)
        act = np.argmax(result["pi"])
        return act

    muzero_obj = None
    def mz(env, obs, info):
        global muzero_obj
        if muzero_obj is None:
            muzero_obj = MuZero(config, model)
        result = muzero_obj.search(State(env, obs=obs), iters=500)
        print(result["pi"].reshape(size, size).round(2))
        act = np.argmax(result["pi"])
        return act

    vepgs_obj = None
    def vepgs(env, obs, info):
        global vepgs_obj
        if vepgs_obj is None:
            vepgs_obj = VEPGS(config, policy1, model)
        result = vepgs_obj.search(State(env, obs=obs), iters=100)
        act = np.argmax(result["pi"])
        return act

    def human(env, obs, info):
        legal_actions = env.legal_actions()
        while True:
            act = int(input("Please type in your action: "))
            if act in [np.argmax(a) for a in legal_actions]:
                break
        return act

    def random(env, obs, info):
        return random_lib.choice(np.arange(size**2)[env.legal_actions()])

    def pn1(env, obs, info):
        # obs (2, 9, 9, 1)
        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(policy1.device)
        with torch.no_grad():
            dist, val = policy1(obs, legal_actions=info["legal_act"][np.newaxis])
        # print dist
        print("Probs:\n", dist.probs.cpu().numpy().reshape(size, size).round(2))
        act = dist.logits.argmax(-1).cpu().numpy()[0]
        return act

    def pn2(env, obs, info):
        # obs (2, 9, 9, 1)
        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(policy1.device)
        with torch.no_grad():
            dist, val = policy2(obs, legal_actions=info["legal_act"][np.newaxis])
        act = dist.logits.argmax(-1).cpu().numpy()[0]
        return act

    # Simulate
    players = [mcts_rust, mcts]
    num_games = 10
    render = True
    num_victories_first = 0
    print(f"Simulating games: {players[0].__name__.upper()} vs {players[1].__name__.upper()}!")
    for i in trange(num_games):
        obs, info = env.reset()

        done = False
        black_turn = True
        while not done:
            if render:
                env.render()

            # Action
            import time
            start = time.time()
            if black_turn:
                act = players[0](env, obs, info)
            else:
                act = players[1](env, obs, info)
            end = time.time()
            print(f"Time: {end - start:.02f} s")

            obs, rew, done, info = env.step(act)

            black_turn = not black_turn

        if render:
            env.render()
        black_turn = not black_turn
        if black_turn:
            num_victories_first += rew

    print(f"Winrate {players[0].__name__.upper()}: {num_victories_first / num_games :.02f}")

    # Close
    if mcts_obj is not None:
        mcts_obj.close()
    if az_obj is not None:
        az_obj.close()
