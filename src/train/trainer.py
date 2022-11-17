import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from copy import deepcopy
from itertools import compress

import os
import gym
import time
import random
from src.debug.elo import update_ratings
from src.debug.util import measure_time
from src.env.envs import Envs
from hexgame import RustEnvs
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
        num_workers = config["num_cpus"]
        num_envs_per_worker = config["num_envs_per_worker"]
        size = config["env"].size
        self.envs = RustEnvs(num_workers, num_envs_per_worker, core_pinning=False, gamma=config["gamma"], max_len=config["max_len"], size=size)
        self.num_envs = config["num_envs"]
        self.policy = None
        self.log = Logger(config, path=f'{PROJECT_PATH}/src/scripts/log/')
        self.save_paths = []
        self.elos = [0]
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config["use_amp"])
        self.scalar_loss = nn.MSELoss()

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
            eps, sample_time = measure_time(lambda: self.sample_sync())
            self.log("sample_time", sample_time)

            _, update_time = measure_time(lambda: self.update(eps))
            self.log("update_time", update_time)

            # Self play test
            if self.config["num_players"] > 1:
                (win_rate, elo), eval_time = measure_time(lambda: self.evaluate())
                self.log("eval_time", eval_time)
                self.log("win_rate", win_rate)
                self.log("elo", elo)

            # Logging
            #self.log.update(eps.metrics)
            print(self.log)
            if self.config["log_to_file"]:
                self.log.to_file()
            if self.config["log_to_writer"]:
                self.log.to_writer(iter)
            self.checkpoint(iter)

    def checkpoint(self, iter):
        last_path = f'{PROJECT_PATH}/checkpoints/policy_{str(self.config["env"]).lower()}_{self.__class__.__name__.lower().replace("trainer","")}_iter={iter}_{self.config["log_main_metric"]}={self.log[self.config["log_main_metric"]]:.0f}.pt'
        self.save_paths.append(last_path)
        if len(self.save_paths) > self.config["num_checkpoints"]:
            path = self.save_paths.pop(0)
            if os.path.isfile(path):
                os.remove(path)
            else:
                print("Warning: %s checkpoint not found" % path)
        self.save(path=last_path)

    @abstractmethod
    def update(self, sample_batch):
        pass

    def sample_sync(self):
        obs, info = self.envs.reset()

        for _ in range(self.config["sample_len"]):
            act, dist = self.get_action(self.policy, obs, info)
            obs, rew, done, info = self.envs.step(act, num_waits=self.num_envs)

        # Sync
        self.envs.reset()
        eps = self.envs.get_episodes()

        return eps

    def get_action(self, policy, obs, info, use_best=False):
        self.policy.eval()
        legal_act = info["legal_act"]

        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        with torch.inference_mode():
            dist = policy.get_dist(obs, legal_actions=legal_act)
        if use_best:
            act = dist.logits.argmax(-1)
        else:
            act = dist.sample()
        act = act.cpu().numpy()
        dist = dist.logits.cpu().numpy()

        if "eid" in info:
            eid = info["eid"]
            act = [(eid[i], act[i]) for i in range(len(act))]

        return act, dist

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
            act, _ = self.get_action(obs[np.newaxis], envs=[deepcopy(env)], use_best=True, legal_actions=[env.legal_actions()])
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
        # if len(self.save_paths) > 0:
        #old_policy.load(self.save_paths[-1])
        # else:
        for layer in old_policy.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        # Parameters
        num_games = self.config["self_play_num_eval_games"]

        # Evaluate in parallel
        win_count = self.play_other(old_policy, num_games)
        win_rate = win_count/num_games * 100.0
        if win_rate > self.config["self_play_update_win_rate"]:
            last_elo = self.elos[-1]
            elo, _ = update_ratings(last_elo, last_elo, num_games, win_count, K=self.config["self_play_elo_k"])
            self.elos.append(elo)
        else:
            self.policy = old_policy
            elo = self.elos[-1]
        return win_rate, elo

    def play_other(self, other_policy, num_games):
        win_count = 0

        for _ in range(int(num_games/self.config["num_envs"])):
            obs, info = self.envs.reset()
            eid = info["eid"]

            for i in range(self.config["max_len"]):
                # Infos
                pid = info["pid"]

                # Current policy
                if np.sum(pid == 0) == 0:
                    act_self = []
                else:
                    obs_self = obs[pid == 0]
                    info_self = {k: v[pid == 0] for k, v in info.items()}
                    act_self, _ = self.get_action(self.policy, obs_self, info_self)

                #  Old policy
                if np.sum(pid == 1) == 0:
                    act_other = []
                else:
                    obs_other = obs[pid == 1]
                    info_other = {k: v[pid == 1] for k, v in info.items()}
                    act_other, _ = self.get_action(other_policy, obs_other, info_other)
                act = act_self + act_other

                # Step
                obs, rew, done, info = self.envs.step(act, num_waits=len(act))

                # Check winner
                for i in range(len(done)):
                    if done[i]:
                        #eid[i]
                        if pid[i] == 0:
                            win_count += rew[i]

                # Remove all env that are finished
                eid = info["eid"]
                eid = eid[~done]
                if len(eid) == 0:
                    break

        return win_count

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
