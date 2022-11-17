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
from src.train.trainer import Trainer
from hexgame import RustEnvs
from src.train.log import Logger

from torch.multiprocessing import freeze_support
from src.train.processer import post_processing
from tabulate import tabulate
from src.env.sample_batch import SampleBatch
from multiprocessing import Pool
import pathlib

PROJECT_PATH = pathlib.Path(__file__).parent.parent.parent.absolute().as_posix()


class PythonTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)
        self.envs = Envs(config)

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
                (win_rate, elo), eval_time = measure_time(lambda: self.evaluate())
                self.log("eval_time", eval_time)
                self.log("win_rate", win_rate)
                self.log("elo", elo)

            # Logging
            self.log.update(sample_batch.metrics)
            print(self.log)
            if self.config["log_to_file"]:
                self.log.to_file()
            if self.config["log_to_writer"]:
                self.log.to_writer(iter)
            self.checkpoint(iter)

    def get_sample_batch(self):
        sample_batch = SampleBatch(self.config)
        obs, info = self.envs.reset()
        pid = info["pid"]
        legal_act = info["legal_act"]

        for _ in range(self.config["sample_len"]):
            act, dist = self.get_action(obs, legal_actions=legal_act)
            obs_next, rew, done, info = self.envs.step(act)

            sample_batch.append(obs, act, rew, done, dist, pid, legal_act)
            pid = info["pid"]
            legal_act = info["legal_act"]
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
            obs, info = self.envs.reset()
            legal_act = info["legal_act"]
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
                act_calc, _ = self.get_action(obs[~dones], env_list=env_list, legal_actions=legal_act[~dones])
                act[~dones] = act_calc
                act_other = [np.arange(self.config["env"].size**2)[x][0] for x in list(compress(legal_act, dones))]
                act[dones] = act_other
                obs_next, rew, done, info = self.envs.step(act)

                # Add new information
                legal_act = info["legal_act"]
                if i % 2 == id:
                    rews += rew * (1 - dones)
                dones |= done
                obs = obs_next

                if dones.all():
                    break

            win_count += np.sum(rews == 1)
        return win_count

    def __enter__(self):
        freeze_support()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.envs.close()
        self.log.close()
