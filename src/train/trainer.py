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
            sample_batch, sample_time = measure_time(lambda: self.get_sample_batch())
            self.log("sample_time", sample_time)

            _, update_time = measure_time(lambda: self.update(sample_batch))
            self.log("update_time", update_time)

            # Self play test
            if self.config["num_players"] > 1:
                if iter > 0:
                    (win_rate, elo), eval_time = measure_time(lambda: self.evaluate())
                    self.log("eval_time", eval_time)
                    self.log("win_rate", win_rate)
                else:
                    elo = 0
                    self.log("win_rate", 0.5)
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

    def get_sample_batch(self):
        sample_batch = SampleBatch(self.config)
        obs, legal_act = self.envs.reset()

        for _ in range(self.config["sample_len"]):
            act, dist = self.get_action(obs, legal_actions=legal_act)
            obs_next, rew, done, info = self.envs.step(act)

            pid, legal_act = info
            sample_batch.append(obs, act, rew, done, dist, pid)
            obs = obs_next

        sample_batch.set_last_obs(obs)
        sample_batch = post_processing(self.policy, sample_batch, self.config)
        return sample_batch

    def get_action(self, obs, env_list=None, use_best=False, legal_actions=None):
        self.policy.eval()
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
        #old_policy.load(self.save_paths[-1])
        for layer in old_policy.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        # Parameters
        num_games = int(np.ceil(self.config["self_play_num_eval_games"]/self.config["num_envs"])) * self.config["num_envs"]

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
        curr_policy = deepcopy(self.policy)
        win_count = 0

        for iter in range(int(num_games/self.config["num_envs"])):
            obs, legal_act = self.envs.reset()
            rews = np.zeros(self.config["num_envs"])
            dones = np.full(self.config["num_envs"], False)
            id = iter % 2

            for i in range(self.config["eval_len"]):
                # Set current policy
                if i % 2 == id:
                    self.policy = curr_policy
                else:
                    self.policy = other_policy

                # Get all current envs that are not done
                env_list = self.envs.get_all_env()
                env_list = [env_list[i] for i in range(len(env_list)) if not dones[i]]

                # If envs finished -> action does not matter (less calculations)
                act = np.zeros(self.config["num_envs"], dtype=np.int32)
                act_calc, _ = self.get_action(obs[~dones], env_list=env_list, legal_actions=list(compress(legal_act, ~dones)))
                act[~dones] = act_calc
                act_other = [x[0] for x in list(compress(legal_act, dones))]
                act[dones] = act_other
                obs_next, rew, done, info = self.envs.step(act)

                # Add new information
                _, legal_act = info
                if i % 2 == id:
                    rews += rew * (1 - dones)
                dones |= done
                obs = obs_next

                if dones.all():
                    break

            win_count += np.sum(rews == 1)
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
