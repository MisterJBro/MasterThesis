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
        result = super().search(iters)
        qvals = result["q"]
        vals = result["v"]

        if self.config["search_return_adv"]:
            max_visits = np.max([child.num_visits for child in self.root.children])
            adv = qvals - self.root.val
            adv = adv / (np.abs(np.max(adv)) + 1e-8)
            pi = (100 + max_visits) * 0.005 * adv
        else:
            pi = self.root.get_normalized_visit_counts(self.config["num_acts"])

        return {
            "q": qvals,
            "v": vals,
            "pi": pi,
        }

    def simulate(self, node):
        # Terminals have zero value
        if node.is_terminal():
            return np.array(0)

        prob, val = self.eval_fn(node)
        node.priors = prob[node.get_legal_actions()]
        return -val

    def set_root(self, state):
        self.root = DirichletNode(state, eps=self.config["dirichlet_eps"], noise=self.config["dirichlet_noise"])
        if state is not None:
            prob, val = self.eval_fn(self.root)
            self.root.priors = prob[self.root.get_legal_actions()]
            self.root.val = val

    def eval_fn(self, node):
        self.eval_channel.send({
            "obs": node.state.obs,
            "ind": self.idx,
        })
        msg = self.eval_channel.recv()
        return msg["prob"], msg["val"]
