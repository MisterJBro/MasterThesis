import numpy as np
from src.search.mcts.core import Tree
from src.search.node import PUCTNode, DirichletNode
from src.search.state import ModelState


class MZTree(Tree):
    """ Search Tree presentation for MuZero. """

    def __init__(self, obs, num_acts, eval_channel, config, idx=0):
        self.config = config
        self.num_acts = num_acts
        self.eval_channel = eval_channel
        self.idx = idx
        self.num_players = config["num_players"]
        self.NodeClass = PUCTNode
        self.expl_coeff = config["puct_c"]
        self.set_root(obs)

    def search(self, iters=1_000):
        qvals = super().search(iters)
        if self.config["tree_output_qvals"]:
            return qvals
        return self.get_normalized_visit_counts()

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
        abs = node.parent.state.abs
        act = node.action
        next_abs, rew, val, prob = self.eval_abs(abs, act)
        node.state = ModelState(next_abs, rew=rew)
        node.priors = prob
        return val

    def set_root(self, state):
        self.root = DirichletNode(state, eps=self.config["dirichlet_eps"], noise=self.config["dirichlet_noise"])
        if self.root.state is not None:
            abs, probs = self.eval_obs(self.root.state.obs)
            self.root.state.abs = abs
            self.root.priors = probs

    def eval_obs(self, obs):
        self.eval_channel.send({
            "obs": obs,
            "ind": self.idx,
        })
        msg = self.eval_channel.recv()
        return msg["abs"], msg["prob"]

    def eval_abs(self, abs, act):
        self.eval_channel.send({
            "abs": abs,
            "act": act,
            "ind": self.idx,
        })
        msg = self.eval_channel.recv()
        return msg["abs"], msg["rew"], msg["val"], msg["prob"]
