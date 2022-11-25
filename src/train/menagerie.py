import os

class PolicyInfo:
    """Meta Information about a policy """

    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        self.elo = 0

    def __str__(self):
        return self.name

class Menagerie:
    """ Menages all policies """

    def __init__(self, config, *args, **kwargs):
        self.config = config
        self.policies = []

    def sample(self):
        pass
