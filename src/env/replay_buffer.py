import numpy as np


class ReplayBuffer:
    """ Replay Buffer for Rust Episodes """

    def __init__(self, config):
        self.capacity = config["replay_buffer_capacity"]
        self.episodes = []
        self.steps = 0

    def add(self, eps):
        while self.steps > self.capacity:
            old_eps = self.episodes.pop(0)
            self.steps -= len(old_eps)
        self.episodes.append(eps)

    def extend(self, eps_list):
        for eps in eps_list:
            self.add(eps)

    def clear(self):
        self.episodes = []
        self.steps = 0

    def sample(self, num_steps):
        batch = []
        batch_steps = 0
        num_steps = min(num_steps, self.steps)

        indices = np.random.choice(len(self.episodes), size=len(self.episodes), replace=False)
        i = 0
        while batch_steps < num_steps:
            eps = self.episodes[indices[i]]
            batch.append(eps)
            batch_steps += len(eps)
            i += 1
        return batch
