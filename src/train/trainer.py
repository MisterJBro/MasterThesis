import numpy as np
import torch
from abc import ABC, abstractmethod
from copy import deepcopy

import gym
import time
import random
from src.debug.elo import update_ratings
from src.env.envs import Envs
from src.train.log import Logger

from torch.multiprocessing import freeze_support
from src.train.processer import post_processing
from tabulate import tabulate
from src.env.sample_batch import SampleBatch
from multiprocessing import Pool
import pathlib

PROJECT_PATH = pathlib.Path(__file__).parent.parent.parent.absolute().as_posix()


class Trainer(ABC):
    def __init__(self, config):
        # RNG seed
        seed = config["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.config = config
        self.device = config["device"]
        self.envs = Envs(config)
        self.policy = None
        self.log = Logger(config, path=f'{PROJECT_PATH}/src/scripts/log/')
        self.eval_pool = Pool(self.config["num_cpus"])
        self.elos = [0]

        print(tabulate([
            ['Environment', config["env"]],
            ['Obs shape', config["obs_dim"]],
            ['Actions num', config["num_acts"]],
            ['CPU count', config["num_cpus"]],
        ], colalign=("left", "right")))
        print()

    def train(self):
        for iter in range(self.config["train_iters"]):
            self.log.clear()
            self.log("iter", iter)

            # Main
            sample_batch = self.get_sample_batch()
            self.update(sample_batch)

            # Self play test
            if self.config["num_players"] > 1:
                if iter > 0:
                    win_rate, elo = self.evaluate()
                    self.log("win_rate", win_rate)
                else:
                    elo = 0
                self.log("elo", elo)

            # Logging
            self.log.update(sample_batch.metrics)
            print(self.log)
            if self.config["log_to_file"]:
                self.log.to_file()
            if self.config["log_to_writer"]:
                self.log.to_writer(iter)
            self.checkpoint(iter)

    def checkpoint(self, iter):
        last_path = f'{PROJECT_PATH}/checkpoints/policy_{str(self.config["env"]).lower()}_{self.__class__.__name__.lower().replace("trainer","")}_iter={iter}_metric={self.log[self.config["log_main_metric"]]:.0f}.pt'
        self.log.save_paths.append(last_path)
        self.save(path=last_path)

    @abstractmethod
    def update(self, sample_batch):
        pass

    def get_sample_batch(self):
        sample_batch = SampleBatch(self.config)
        obs, legal_act = self.envs.reset()

        for _ in range(self.config["sample_len"]):
            act, dist = self.get_action(obs, envs=self.envs, legal_actions=legal_act)
            obs_next, rew, done, info = self.envs.step(act)

            pid, legal_act = info
            sample_batch.append(obs, act, rew, done, dist, pid)
            obs = obs_next

        sample_batch.set_last_obs(obs)
        sample_batch = post_processing(self.policy, sample_batch, self.config)
        return sample_batch

    def get_action(self, obs, envs=None, use_best=False, legal_actions=None):
        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            dist = self.policy.get_dist(obs, legal_actions=legal_actions)
        if use_best:
            act = dist.logits.argmax(-1)
        else:
            act = dist.sample()

        return act.cpu().numpy(), dist.logits.cpu().numpy()

    def test(self, render=True):
        if isinstance(self.config["env"], str):
            env = gym.make(self.config["env"])
        else:
            env = self.config["env"]
        rews = []
        if render:
            input('Press any key to continue...')

        obs = env.reset()
        for _ in range(self.config["test_len"]):
            if render:
                deepcopy(env).render()
                time.sleep(0.1)
            act, _ = self.get_action(obs[np.newaxis], envs=[deepcopy(env)], use_best=True, legal_actions=[env.available_actions()])
            obs, rew, done, _ = env.step(act[0])
            rews.append(rew)

            if done:
                obs = env.reset()
                break
        print(f'Undiscounted return: {np.sum(rews)}')
        env.close()

    def evaluate(self):
        # Get last policy
        old_policy = deepcopy(self.policy)
        old_policy.load(self.log.save_paths[-1])

        # Parameters
        p = self.eval_pool
        num_worker = self.config["num_cpus"]
        num_games = max(self.config["num_eval_games"], num_worker)
        num_games -= num_games % num_worker
        env = self.config["env"]
        sample_len = self.config["sample_len"]

        # Evaluate in parallel
        win_count = sum(p.starmap(evaluate, [(int(num_games / num_worker), env, self.policy, old_policy, sample_len) for _ in range(num_worker)]))
        win_rate = win_count/num_games * 100.0
        if win_rate > self.config["min_win_rate_to_update"]:
            last_elo = self.elos[-1]
            elo, _ = update_ratings(last_elo, last_elo, num_games, win_count, K=30)
            self.elos.append(elo)
        else:
            self.policy = old_policy
            elo = self.elos[-1]
        return win_rate, elo

    def save(self, path=None):
        if path is not None:
            self.policy.save(path=path)
        else:
            path = f'{PROJECT_PATH}/checkpoints/policy_{self.config["env"]}_{self.__class__.__name__.lower()}.pt'
            self.policy.save(path=path)

    def load(self, path=None):
        if path is not None:
            self.policy.load(path=path)
        else:
            path = f'{PROJECT_PATH}/checkpoints/policy_{self.config["env"]}_{self.__class__.__name__.lower()}.pt'
            self.policy.load(path=path)

    def __enter__(self):
        freeze_support()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.envs.close()
        self.log.close()
        self.eval_pool.close()

def nn(env, obs, policy):
    obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(policy.device)
    with torch.no_grad():
        dist = policy.get_dist(obs, legal_actions=[env.available_actions()])
    act = dist.sample().cpu().numpy()[0]
    return act

# Self play evaluation
def evaluate(num_games_per_worker, env, agent1, agent2, sample_len):
    env = deepcopy(env)
    obs_first = None

    win_count = 0
    for n in range(num_games_per_worker):
        # Get player order, switch every game
        pid = n % 2
        eid = (n+1) % 2

        # Play two times with the same seed for more accurate winrate prediction (so no one can have a luck based board advantage)
        if n % 2 == 0:
            obs_first = env.reset()
            env2 = deepcopy(env)
        else:
            env = env2
        obs = obs_first

        for i in range(sample_len):
            # Get action
            if i % 2 == pid:
                act = nn(env, obs, agent1)
            else:
                act = nn(env, obs, agent2)

            # Simulate one step and get new obs
            obs, rew, done, info = env.step(act)

            if done:
                if i % 2 == pid:
                    win_count += rew
                break
    return win_count