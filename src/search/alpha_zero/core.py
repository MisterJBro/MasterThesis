import numpy as np
from src.search.mcts.core import MCTSCore
from src.search.node import PUCTNode, DirichletNode


class AZCore(MCTSCore):
    """ Core Algorithm for AlphaZero worker. """

    def __init__(self, config, state, eval_channel, idx=0):
        self.config = config
        self.eval_channel = eval_channel
        self.idx = idx
        self.num_players = config["num_players"]
        self.NodeClass = PUCTNode
        self.expl_coeff = config["puct_c"]
        self.set_root(state)

    def search(self, iters):
        qvals = super().search(iters, default=self.root.val)
        if self.config["tree_output_qvals"]:
            max_visits = np.max([child.num_visits for child in self.root.children])
            adv = qvals - self.root.val
            adv = (adv - np.min(adv)) / (np.max(adv) - np.min(adv))
            adv = 2*adv - 1
            return (100 + max_visits) * 0.1 * adv
        return self.get_normalized_visit_counts()

    def simulate(self, node):
        # Terminals have zero value
        if node.state.is_terminal():
            return np.array(0)

        prob, val = self.eval_fn(node)
        node.priors = prob
        return val

    def set_root(self, state):
        self.root = DirichletNode(state, eps=self.config["dirichlet_eps"], noise=self.config["dirichlet_noise"])
        if state is not None:
            prob, val = self.eval_fn(self.root)
            self.root.priors = prob
            self.root.val = val

    def eval_fn(self, node):
        self.eval_channel.send({
            "obs": node.state.obs,
            "ind": self.idx,
        })
        msg = self.eval_channel.recv()
        return msg["prob"], msg["val"]
