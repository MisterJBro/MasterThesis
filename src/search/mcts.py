import numpy as np
from tree import TreeWorker
from state import State
from copy import deepcopy
from multiprocessing import Pipe, freeze_support

class MCTS:
    """ Monte Carlo Tree Search, with root parallelization."""
    def __init__(self, config):
        self.config = config
        self.num_workers = config["num_trees"]
        self.num_iters = config["mcts_iters"]

        # Create parallel tree workers
        pipes = [Pipe() for _ in range(self.num_workers)]
        self.channels = [p[0] for p in pipes]
        self.num_iters_worker = int(self.num_iters/self.num_workers)
        self.rest_iters = (self.num_iters % self.num_workers) + self.num_iters_worker
        self.workers = [
            TreeWorker(self.rest_iters if i == self.num_workers-1 else self.num_iters_worker, config, pipes[i][1])
            for i in range(self.num_workers)
        ]
        for w in self.workers:
            w.start()

    def search(self, state, iters=None):
        for c in self.channels:
            c.send({
                "command": "search",
                "state": deepcopy(state),
                "iters": iters,
            })
        msg = np.stack([c.recv() for c in self.channels])

        qvals = np.mean(msg, axis=0)
        return qvals

    def close(self):
        for c in self.channels:
            c.send({"command": "close"})
        for w in self.workers:
            w.join()


if __name__ == "__main__":
    from gym_tictactoe.env import TicTacToeEnv
    freeze_support()

    config = {
        "uct_c": np.sqrt(2),
        "mcts_iters": 10_000,
        "num_trees": 4,
        "bandit_policy": "uct",
        "num_players": 2,
    }

    env = TicTacToeEnv()
    mcts = MCTS(config)
    obs = env.reset()
    qvals = mcts.search(State(env), iters=10)

    done = False
    while not done:
        import time
        start = time.time()
        qvals = mcts.search(State(env))

        act = env.available_actions()[np.argmax(qvals)]
        obs, reward, done, info = env.step(act)
        print(f"Reward: {reward:0.2f}  Time: {time.time() - start:0.2f}s")

        env.render()
    env.close()
    mcts.close()