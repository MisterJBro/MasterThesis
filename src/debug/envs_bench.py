from multiprocessing import freeze_support
from src.env.hex import HexEnv
from src.env.envs import Envs
from src.train.config import create_config
import time
import random


if __name__ == '__main__':
    # Init
    freeze_support()
    size = 9
    env = HexEnv(size)
    config = create_config({
        "env": env,
        "num_cpus": 3,
        "num_envs": 12,
        "sample_len": 1_000,
    })
    envs = Envs(config)

    def python_mp():
        time_needed = 0
        start = time.time()
        obs, legal_act = envs.reset()
        end = time.time()
        time_needed += end - start

        for _ in range(config["sample_len"]):
            act = [random.choice(legal_act[i]) for i in range(len(legal_act))]
            start = time.time()
            obs_next, rew, done, info = envs.step(act)
            end = time.time()
            time_needed += end - start

            pid, legal_act = info
            obs = obs_next

        return time_needed

    # Measure time
    time_needed = python_mp()
    print(f"Python time taken: {time_needed:.02f}s")

    envs.close()
