import os
import random
import time
import humanize
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from hexgame import RustEnvs
from tabulate import tabulate
from torch.multiprocessing import freeze_support

import gym
from src.debug.elo import update_ratings
from src.debug.util import measure_time, seed_all
from src.train.log import Logger
from src.train.menagerie import Menagerie, PolicyMapping
from src.train.config import to_yaml


class Trainer(ABC):
    """ Trainer Base Class """

    def __init__(self, config, policy, **kwargs):
        # Init attributes
        seed_all(config["seed"])
        self.config = config
        self.policy = policy
        self.__dict__.update(kwargs)

        # Init Envs
        self.num_envs = config["num_envs"]
        self.envs = RustEnvs(
            config["num_workers"],
            config["num_envs_per_worker"],
            core_pinning=config["core_pinning"],
            gamma=config["gamma"],
            max_len=config["max_len"],
            size=config["env"].size,
        )

        # Under experiment path get all folder names
        dir_names = [name for name in os.listdir(config["experiment_path"]) if os.path.isdir(config["experiment_path"] + name) and name.isdigit()]
        print(dir_names)
        if len(dir_names) > 0:
            new_name = max([int(name) for name in dir_names ]) + 1
        else:
            new_name = 0
        self.config["experiment_path"] = self.config["experiment_path"] + str(new_name) + '/'
        os.makedirs(self.config["experiment_path"])
        to_yaml(self.config, self.config["experiment_path"] + "config.yaml")

        # Others
        self.device = config["device"]
        self.log = Logger(config)
        self.menagerie = Menagerie(config, policy, self.log)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config["use_amp"])

        # Print information
        print(tabulate([
            ['Environment', config["env"].__class__.__name__],
            ['Policy params', humanize.intword(self.policy.get_num_params())],
            ['Obs shape', config["obs_dim"]],
            ['Actions', config["num_acts"]],
            ['Workers', config["num_workers"]],
        ], colalign=("left", "right")))
        print()

    def train(self):
        self.log("iter", 0)
        eps, sample_time = measure_time(lambda: self.sample_sync())
        self.log("sample_t", sample_time, 's', show=False)

        for iter in range(self.config["train_iters"]):
            self.log("iter", iter+1)

            # Main
            #self.pre_update()
            _, update_time = measure_time(lambda: self.update(eps))
            self.log("update_t", update_time, 's')

            eps, sample_time = measure_time(lambda: self.sample_sync())
            self.log("sample_t", sample_time, 's')

            # Self play test
            #self.policy.eval()
            #if self.config["num_players"] > 1:
            #    (win_rate, elo), eval_time = measure_time(lambda: self.evaluate())
            #    self.log("eval_t", eval_time, 's')
            #    self.log("win_rate", win_rate, '%')
            #    self.log("elo", elo)

            # Logging
            print(self.log)
            if self.config["log_to_file"]:
                self.log.to_file()
            if self.config["log_to_writer"]:
                self.log.to_writer()

    @abstractmethod
    def update(self, eps):
        pass

    def sample_sync(self):
        self.policy.eval()
        policies = self.menagerie.sample()
        map = PolicyMapping(len(policies), self.num_envs)

        # Metrics
        num_games = 0
        games = {}
        last_pol_id = np.zeros(self.num_envs, dtype=np.int32)

        # Reset
        obs, info = self.envs.reset()

        for _ in range(self.config["sample_len"]):
            act, eid, dist, pol_id = self.get_different_actions(policies, map, obs, info)
            last_pol_id[eid] = pol_id
            obs, rew, done, info = self.envs.step(act, eid, dist, pol_id, num_waits=self.num_envs)

            # Check dones
            for i in info["eid"]:
                if done[i]:
                    num_games += 1
                    if last_pol_id[i] == 0:
                        win_base = (rew[i] + 1) / 2
                    else:
                        win_base = 1 - (rew[i] + 1) / 2

                    g = tuple(map[i])
                    if g not in games:
                        games[g] = {
                            "num": 1,
                            "win_base": win_base,
                        }
                    else:
                        games[g]["num"] += 1
                        games[g]["win_base"] += win_base

                    map.update(i)

        # Sync
        self.log("games", games, show=False)
        self.log("num_games", num_games)
        self.envs.reset()
        eps = self.envs.get_episodes()
        self.menagerie.update()

        return eps

    def get_action(self, policy, obs, info, use_best=False):
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
            return act, dist, info["eid"]

        return act, dist

    # Get actions from different policies according to some mapping of policy to env
    def get_different_actions(self, policies, mapping, obs, info, use_best=False):
        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        legal_act = info["legal_act"]
        eid = info["eid"]
        pid = info["pid"]
        pol_id = mapping[eid, pid]

        # CUDA inference (async)
        eids, dists, pol_ids = [], [], []
        for p, policy in enumerate(policies):
            is_p = pol_id == p
            if np.sum(is_p) > 0:
                obs_p = obs[is_p]
                legal_act_p = legal_act[is_p]
                with torch.inference_mode():
                    dist_p = policy.get_dist(obs_p, legal_actions=legal_act_p)
                dists.append(dist_p)
                eids.append(eid[is_p])
                pol_ids.append(np.full(np.sum(is_p), p))

        # Build actions
        act = []
        eid = []
        pol_id = []
        dist = []
        for eid_p, dist_p, pol_id_p in zip(eids, dists, pol_ids):
            if use_best:
                act_p = dist_p.logits.argmax(-1)
            else:
                act_p = dist_p.sample()
            act_p = [a.cpu().numpy().item() for a in act_p]
            act += act_p
            eid.append(eid_p)
            pol_id.append(pol_id_p)
            dist.append(dist_p.logits.cpu().numpy())
        return act, np.concatenate(eid, 0), np.concatenate(dist, 0), np.concatenate(pol_id, 0)

    def test(self, render=True):
        if isinstance(self.config["env"], str):
            env = gym.make(self.config["env"])
        else:
            env = self.config["env"]
        rews = []
        if render:
            input('Press any key to continue...')

        obs = env.reset()
        for _ in range(self.config["max_len"]):
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
        for layer in old_policy.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        # Parameters
        num_games = self.config["sp_num_eval_games"]
        if num_games == 0:
            return 0, 0

        # Evaluate in parallel
        win_count = self.play_other(old_policy, num_games)
        win_rate = win_count/(num_games+1e-9) * 100.0
        if win_rate >= self.config["sp_update_win_rate"]:
            last_elo = self.elos[-1]
            elo, _ = update_ratings(last_elo, last_elo, num_games, win_count, K=self.config["sp_elo_k"])
            self.elos.append(elo)
        else:
            self.policy.load_state_dict(deepcopy(self.last_policy.state_dict()))
            elo = self.elos[-1]
        return win_rate, elo

    def play_other(self, other_policy, num_games):
        win_count = 0
        # Policy mapping
        policies = [self.policy, other_policy]
        policy_mapping = np.zeros((self.num_envs, 2), dtype=np.int32)
        policy_mapping[::2, 1] = 1
        policy_mapping[1::2, 0] = 1

        for _ in range(int(num_games/self.num_envs)):
            obs, info = self.envs.reset()

            for i in range(self.config["max_len"]):
                # Get actions
                act, eid, dist, pol_id  = self.get_different_actions(policies, policy_mapping, obs, info)

                # Step
                next_obs, rew, done, next_info = self.envs.step(act, eid, dist, pol_id, num_waits=self.num_envs)

                # Check winner
                for i in range(len(done)):
                    if done[i]:
                        if policy_mapping[info["eid"][i], info["pid"][i]] == 0:
                            win_count += rew[i]

                # Remove all env that are finished
                info = next_info
                obs = next_obs[~done]
                info = {k: v[~done] for k, v in info.items()}
                if len(info["eid"]) == 0:
                    break

            # Change mapping
            policy_mapping = 1 - policy_mapping

        return win_count

    def __enter__(self):
        freeze_support()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.envs.close()
        self.log.close()
