from multiprocessing import freeze_support
from src.env.hex import HexEnv
from src.env.envs import Envs
from hexgame import RustEnvs
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
    rust_envs = RustEnvs(config["num_cpus"], int(config["num_envs"]/config["num_cpus"]), size)

    def python_mp():
        start = time.time()
        obs, info = envs.reset()
        pid = info["pid"]
        legal_act = info["legal_act"]

        for _ in range(config["sample_len"]):
            act = [random.choice(legal_act[i]) for i in range(len(legal_act))]
            obs_next, rew, done, info = envs.step(act)

            pid = info["pid"]
            legal_act = info["legal_act"]
            obs = obs_next

        end = time.time()
        return end-start

    def rust():
        start = time.time()
        obs, info = rust_envs.reset()
        pid = info["pid"]
        legal_act = info["legal_act"]

        for _ in range(config["sample_len"]):
            act = [random.choice(legal_act[i]) for i in range(len(legal_act))]
            obs_next, rew, done, info = rust_envs.step(act)

            pid = info["pid"]
            legal_act = info["legal_act"]
            obs = obs_next

        end = time.time()
        return end-start

    # Measure time
    time_needed = python_mp()
    print(f"Python time taken: {time_needed:.02f}s")
    time_needed = rust()
    print(f"Rust time taken: {time_needed:.02f}s")

    envs.close()
    rust_envs.close()
