import numpy as np
from torch.multiprocessing import freeze_support
from src.train.config import create_config
from src.env.hex import HexEnv
from src.search.mcts.mcts import MCTS
from src.search.state import State


if __name__ == '__main__':
    freeze_support()

    # Init for algos
    size = 5
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
    #policy = PendulumPolicy(config)
    #policy.load("checkpoints/policy_pdlm_pgtrainer.pt")
    #model = ValueEquivalenceModel(config)
    #model.load("checkpoints/ve_model.pt")

    # Algorithms
    mcts = MCTS(config)
    obs = env.reset()

    done = False
    is_player = True
    while not done:
        avacts = env.available_actions()
        print([np.argmax(a) for a in avacts])
        if is_player:
            while True:
                act = int(input("Please type in your action: "))
                if act > 0 and act < size * size:
                    act = env.available_actions()[act]
                    break
        else:
            dist = mcts.search(State(env, obs=obs), iters=100)
            act = env.available_actions()[np.argmax(dist)]

        obs, reward, done, _ = env.step(act)

        env.render()
        is_player = not is_player

    # Close
    mcts.close()
