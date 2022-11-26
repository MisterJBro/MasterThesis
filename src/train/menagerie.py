import os
import random
import numpy as np
from copy import deepcopy


class PolicyInfo:
    """ Meta Information about a policy """

    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        self.elo = 0

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
        self.policies = [self.policy]
        self.infos = []

    def sample(self):
        """ Sample policies for use in self-play """
        if len(self.infos[:-1]) > 0:
            # Adding new policies
            if len(self.policies) < self.config["sp_sampled_policies"]:
                new_policy = deepcopy(self.policy)
                new_policy.eval()
                self.policies.append(new_policy)

            # Sample and load
            infos = random.sample(self.infos[:-1], max(0, len(self.policies) - 2))
            for i in range(2, len(self.policies)):
                self.policies[i].load(path=infos[i-2].path)
        return self.policies

    def update(self):
        """ Updates menagerie with last policy """
        if self.config["sp_sampled_policies"] > 1:
            if len(self.policies) == 1:
                self.last_policy = deepcopy(self.policy)
                self.last_policy.eval()
                self.policies.append(self.last_policy)
            else:
                self.last_policy.load_state_dict(deepcopy(self.policy.state_dict()))
        self.checkpoint()

    def checkpoint(self):
        name = f'p_{self.log["iter"]["value"][-1]}.pt'#_{self.config["log_main_metric"]}={self.log[self.config["log_main_metric"]]["value"][-1]:.0f}.pt'
        path = self.config["experiment_path"] + name
        self.infos.append(PolicyInfo(path))

        #if len(self.save_paths) > self.config["num_checkpoints"]:
        #    last_path = self.save_paths.pop(0)
        #    if os.path.isfile(last_path):
        #        os.remove(last_path)
        #    else:
        #        print("Warning: %s checkpoint not found" % last_path)
        self.policy.save(path=path)

    def elo(self):
        pass

