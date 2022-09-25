import numpy as np
from src.search.alpha_zero.core import AZCore
from src.search.node import DirichletNode
from src.search.state import ModelState


class MZCore(AZCore):
    """ Core Algorithm for for MuZero worker. """

    def __init__(self, config, state, num_acts, eval_channel, idx=0):
        super().__init__(config, state, eval_channel, idx=idx)
        self.num_acts = num_acts

    def expand(self, node):
        if node.state is None:
            return node

        # Create new child nodes, lazy init
        actions = np.eye(self.num_acts)
        for action in actions:
            new_node = self.NodeClass(None, action=action, parent=node)
            node.children.append(new_node)

        # Pick child node
        child = np.random.choice(node.children)
        return child

    def simulate(self, node):
        prob, next_abs, rew, val = self.eval_abs(node)
        node.state = ModelState(next_abs, rew=rew)
        node.priors = prob
        return val

    def set_root(self, state):
        self.root = DirichletNode(state, eps=self.config["dirichlet_eps"], noise=self.config["dirichlet_noise"])
        if self.root.state is not None:
            prob, abs, val = self.eval_obs(self.root)
            self.root.priors = prob
            self.root.val = val
            self.root.state.abs = abs

    def eval_obs(self, node):
        self.eval_channel.send({
            "obs": node.state.obs,
            "ind": self.idx,
        })
        msg = self.eval_channel.recv()
        return msg["prob"], msg["abs"], msg["val"]

    def eval_abs(self, node):
        self.eval_channel.send({
            "abs": node.parent.state.abs,
            "act": node.action,
            "ind": self.idx,
        })
        msg = self.eval_channel.recv()
        return msg["prob"], msg["abs"], msg["rew"], msg["val"]
