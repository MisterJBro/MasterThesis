import os
import humanize
from abc import ABC, abstractmethod

import numpy as np
import torch
from hexgame import RustEnvs
from tabulate import tabulate
from torch.multiprocessing import freeze_support

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
        self.log.show()

        for iter in range(self.config["train_iters"]):
            self.log("iter", iter+1)

            # Main
            _, update_time = measure_time(lambda: self.update(eps))
            self.log("update_t", update_time, 's')

            eps, sample_time = measure_time(lambda: self.sample_sync())
            self.log("sample_t", sample_time, 's')
            self.log.show()

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
            act, eid, dist, pol_id = self.get_self_play_actions(policies, map, obs, info)
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

    # Get actions from different policies according to some mapping of policy to env
    def get_self_play_actions(self, policies, mapping, obs, info, use_best=False):
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

    def __enter__(self):
        freeze_support()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.envs.close()
        self.log.close()
