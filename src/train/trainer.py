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
    def __init__(self, config, policy):
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
        self.policy = policy
        self.log = Logger(config, path=f'{PROJECT_PATH}/src/scripts/log/')
        self.save_paths = []
        self.elos = [0]
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config["use_amp"])
        self.scalar_loss = nn.MSELoss()
        self.policies = [self.policy]

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
            self.log("sample_t", sample_time, 's')

            self.pre_update()
            _, update_time = measure_time(lambda: self.update(eps))
            self.log("update_t", update_time, 's')

            # Self play test
            self.policy.eval()
            if self.config["num_players"] > 1:
                (win_rate, elo), eval_time = measure_time(lambda: self.evaluate())
                self.log("eval_t", eval_time, 's')
                self.log("win_rate", win_rate, '%')
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
        dir_path = f'{PROJECT_PATH}/checkpoints/'
        name = f'p_{iter}_{self.config["log_main_metric"]}={self.log[self.config["log_main_metric"]][0]:.0f}.pt'
        next_path = dir_path + name
        self.save_paths.append(next_path)
        if len(self.save_paths) > self.config["num_checkpoints"]:
            last_path = self.save_paths.pop(0)
            if os.path.isfile(last_path):
                os.remove(last_path)
            else:
                print("Warning: %s checkpoint not found" % last_path)
        self.save(path=next_path)

    def pre_update(self):
        self.policy.train()
        if len(self.policies) == 1:
            self.last_policy = deepcopy(self.policy)
            self.last_policy.eval()
            self.policies.append(self.last_policy)
        else:
            self.last_policy.load_state_dict(deepcopy(self.policy.state_dict()))

    @abstractmethod
    def update(self, sample_batch):
        pass

    # Sample Policies for use in Self Play
    def sample_policies(self):
        if len(self.save_paths[:-1]) > 0:
            # Adding new policies
            if len(self.policies) < self.config["sp_sampled_policies"]:
                new_policy = deepcopy(self.policy)
                new_policy.eval()
                self.policies.append(new_policy)

            # Sample
            paths = random.sample(self.save_paths[:-1], max(0, len(self.policies) - 2))
            for i in range(2, len(self.policies)):
                path = paths[i-2]
                self.policies[i].load(path=path)

    def sample_sync(self):
        self.policy.eval()
        self.sample_policies()

        policy_mapping = np.zeros((self.num_envs, 2), dtype=np.int32)
        #if len(self.policies) > 1:
        #    policy_mapping[::2, 1] = 1
        #    policy_mapping[1::2, 0] = 1

        # Metrics
        num_games = 0
        num_wins = np.zeros(len(self.policies), dtype=np.int32)
        last_pol_id = np.zeros(self.num_envs, dtype=np.int32)
        game_hist = {}
        game_rates = {}

        # Reset
        obs, info = self.envs.reset()

        for _ in range(self.config["sample_len"]):
            act, eid, dist, pol_id = self.get_different_actions(self.policies, policy_mapping, obs, info)
            last_pol_id[eid] = pol_id
            # possible act = self.get_different_actions(...)
            # self.envs.step(*act) # python spread operator
            obs, rew, done, info = self.envs.step(act, eid, dist, pol_id, num_waits=self.num_envs)

            # Check dones
            for i in info["eid"]:
                if done[i]:
                    num_games += 1
                    wid = last_pol_id[i]
                    num_wins[wid] += rew[i]

                    # Change policy mapping
                    if policy_mapping[i].sum() == 0:
                        # Randomly insert to either index 0 or 1
                        policy_mapping[i, random.randint(0, 1)] = 1 % len(self.policies)
                    else:
                        policy_mapping[i, policy_mapping[i] > 0] = (policy_mapping[i, policy_mapping[i] > 0] + 1 ) % len(self.policies)

                    if bool(random.getrandbits(1)):
                        policy_mapping[i] = policy_mapping[i][::-1]
                    game_hist[tuple(policy_mapping[i].tolist())] = game_hist.get(tuple(policy_mapping[i].tolist()), 0) + 1
                    game_rates[tuple(policy_mapping[i].tolist())] = game_rates.get(tuple(policy_mapping[i].tolist()), 0) + rew[i]

        #print(game_hist)
        # Sync
        self.log("num_games", num_games)
        self.envs.reset()
        eps = self.envs.get_episodes()

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
