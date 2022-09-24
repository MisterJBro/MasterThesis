import numpy as np
from src.search.mcts.core import Tree
from src.search.node import PUCTNode, DirichletNode


class AZTree(Tree):
    """ Search Tree presentation for Alpha Zero. """

    def __init__(self, state, eval_channel, config, idx=0):
        self.config = config
        self.eval_channel = eval_channel
        self.idx = idx
        self.num_players = config["num_players"]
        self.NodeClass = PUCTNode
        self.expl_coeff = config["puct_c"]
        self.set_root(state)

    def search(self, iters=1_000):
        qvals = super().search(iters)
        if self.config["tree_output_qvals"]:
            return qvals
        return self.get_normalized_visit_counts()

    def simulate(self, node):
        # Terminals have zero value
        if node.state.is_terminal():
            return np.array(0)

        probs, val = self.eval_fn(node.state.obs)
        node.priors = probs
        return val

    def set_root(self, state):
        self.root = DirichletNode(state, eps=self.config["dirichlet_eps"], noise=self.config["dirichlet_noise"])
        if state is not None:
            probs, _ = self.eval_fn(self.root.state.obs)
            self.root.priors = probs

    def eval_fn(self, obs):
        self.eval_channel.send({
            "obs": obs,
            "ind": self.idx,
        })
        msg = self.eval_channel.recv()
        return msg["prob"], msg["val"]
