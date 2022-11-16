import numpy as np
from copy import deepcopy
from torch.multiprocessing import Process


class SampleManager(Process):
    """Manages incoming samples by collecting and processing them for training."""

    def __init__(self, channel, config):
        super(SampleManager, self).__init__()
        self.config = config
        self.channel = channel
        self.trajs = [[] for _ in range(config["num_envs"])]

    def run(self):
        while True:
            msg = self.channel.recv()
            self.collect(msg)

    def collect(self, samples):
        pass

    def process(self, traj):
        pass

