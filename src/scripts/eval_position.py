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
import math


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    freeze_support()

    # Init for algos
    size = 7
    env = HexEnv(size)
    config = create_config({
        "env": env,
        "puct_c": 3.0,
        "uct_c": math.sqrt(2),
        "search_num_workers": 1,
        "search_evaluator_batch_size": 1,
        "dirichlet_eps": 0.0,
        "pgs_lr": 0e-1,
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
    model = ValueEquivalenceModel(config)
    #model.load("checkpoints/m_5x5_10_128.pt")

    # Algorithms /Players
    mcts_obj = None
    def mcts(env, obs, info, iters):
        global mcts_obj
        if mcts_obj is None:
            mcts_obj = MCTS(config)
        result = mcts_obj.search(State(env, obs=obs), iters=iters)
        return result

    az_obj = None
    def az(env, obs, info, iters):
        global az_obj
        if az_obj is None:
            az_obj = AlphaZero(config, policy1)
        result = az_obj.search(State(env, obs=obs), iters=iters)
        return result

    pgs_obj = None
    def pgs(env, obs, info, iters):
        global pgs_obj
        if pgs_obj is None:
            pgs_obj = PGS(config, policy1)
        result = pgs_obj.search(State(env, obs=obs), iters=iters)
        return result

    muzero_obj = None
    def mz(env, obs, info, iters):
        global muzero_obj
        if muzero_obj is None:
            muzero_obj = MuZero(config, model)
        result = muzero_obj.search(State(env, obs=obs), iters=iters)
        return result

    vepgs_obj = None
    def vepgs(env, obs, info, iters):
        global vepgs_obj
        if vepgs_obj is None:
            vepgs_obj = VEPGS(config, policy1, model)
        result = vepgs_obj.search(State(env, obs=obs), iters=iters)
        return result

    def random(env, obs, info):
        return random_lib.choice(np.arange(size**2)[env.legal_actions()])

    def pn(env, obs, info, iters):
        # obs (2, 9, 9, 1)
        obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(policy1.device)
        with torch.no_grad():
            dist, val = policy1(obs, legal_actions=info["legal_act"][np.newaxis])
        probs = dist.probs.cpu().numpy()[0]
        return {"pi": probs, "v": val.cpu().numpy(), "q": probs}

    # Simulate
    player = pgs

    # Setting up position
    render = True
    moves = []
    obs, info = env.reset()
    for m in moves:
        obs, rew, done, info = env.step(m)
    env.render()
    num_victories_first = 0
    print(f"Eval positon with: {player.__name__.upper()}!")

    for iters in [10, 50, 100, 200, 500, 1000]:
        print("Iters: ", iters)
        result = player(env, obs, info, iters)
        result["q"][result["q"] < -1e6] = -np.inf
        print("Q:\n", result["q"].reshape(size, size).round(2))
        print("V: ", result["v"].round(3).item())

    # Close
    if mcts_obj is not None:
        mcts_obj.close()
    if az_obj is not None:
        az_obj.close()
