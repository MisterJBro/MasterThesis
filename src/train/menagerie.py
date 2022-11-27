import os
import random
import numpy as np
from copy import deepcopy
from src.debug.elo import update_ratings


class PolicyInfo:
    """ Meta Information about a policy """

    def __init__(self, path, elo):
        self.path = path
        self.elo = elo

    def get_name(self):
        return os.path.basename(self.path)

    def __str__(self):
        return self.name


class PolicyMapping:
    """ Maps policies to environments """

    def __init__(self, num_policies, num_envs):
        self.num_policies = num_policies
        self.num_envs = num_envs
        self.map = np.zeros((num_envs, 2), dtype=np.int32)
        if num_policies > 1:
            self.map[::2, 1] = 1
            self.map[1::2, 0] = 1

    def __getitem__(self, indices):
        return self.map[indices]

    def update(self, i):
        # Increment id
        if self.map[i].sum() == 0:
            self.map[i, random.randint(0, 1)] = 1 % self.num_policies
        else:
            self.map[i, self.map[i] > 0] = (self.map[i, self.map[i] > 0] + 1 ) % self.num_policies

        # Switch sides
        if bool(random.getrandbits(1)):
            self.map[i] = self.map[i][::-1]

class Menagerie:
    """ Menages all policies """

    def __init__(self, config, policy, log):
        self.config = config
        self.policy = policy
        self.log = log
        self.curr_elo = config["sp_start_elo"]
        self.policies = [self.policy]
        self.infos = []
        self.sampled_infos = []

    def sample(self):
        """ Sample policies for use in self-play """
        self.sampled_infos = []
        if len(self.infos[:-1]) > 0:
            # Adding new policies
            if len(self.policies) < self.config["sp_sampled_policies"]:
                new_policy = deepcopy(self.policy)
                new_policy.eval()
                self.policies.append(new_policy)

            # Sample and load
            indices = random.sample(list(np.arange(len(self.infos[:-1]))), max(0, len(self.policies) - 2))
            for i in range(2, len(self.policies)):
                self.policies[i].load(path=self.infos[indices[i-2]].path)
                self.sampled_infos.append(indices[i-2])
        self.log("sampled_policies", self.sampled_infos, show=False)
        return self.policies

    def update(self):
        """ Updates menagerie with last policy """
        self.elo()
        self.checkpoint()
        self.new_last()

    def elo(self):
        """ Calculates new elo ratings for policies which played games """
        games = self.log["games"]["value"][-1]
        updated_infos = []

        for match_up, data in games.items():
            num_games = data["num"]
            num_wins = data["win_base"]
            opponent = max(match_up)
            if opponent == 0:
                continue
            elif opponent == 1:
                self.curr_elo, self.infos[-1].elo = update_ratings(self.curr_elo, self.infos[-1].elo, num_games, num_wins, K=self.config["sp_elo_k"])
                self.log["elo"]["value"][-1] = self.infos[-1].elo
                updated_infos.append(self.infos[-1])
            else:
                index = self.sampled_infos[opponent-2]
                info = self.infos[index]
                self.curr_elo, info.elo = update_ratings(self.curr_elo, info.elo, num_games, num_wins, K=self.config["sp_elo_k"])
                self.log["elo"]["value"][index] = self.infos[index].elo
                updated_infos.append(info)

        self.log("elo", self.curr_elo)
        for info in updated_infos:
            new_path = info.path.split("_elo_")[0] + f"_elo_{info.elo:.01f}.pt"
            os.rename(info.path, new_path)
            info.path = new_path

    def checkpoint(self):
        name = f'p_{self.log["iter"]["value"][-1]}_elo_{self.curr_elo:.01f}.pt'
        path = self.config["experiment_path"] + name
        self.infos.append(PolicyInfo(path, self.curr_elo))

        #if len(self.save_paths) > self.config["num_checkpoints"]:
        #    last_path = self.save_paths.pop(0)
        #    if os.path.isfile(last_path):
        #        os.remove(last_path)
        #    else:
        #        print("Warning: %s checkpoint not found" % last_path)
        self.policy.save(path=path)

    def new_last(self):
        if self.config["sp_sampled_policies"] > 1:
            if len(self.policies) == 1:
                self.last_policy = deepcopy(self.policy)
                self.last_policy.eval()
                self.policies.append(self.last_policy)
            else:
                self.last_policy.load_state_dict(deepcopy(self.policy.state_dict()))
        #self.curr_elo = self.config["sp_start_elo"]