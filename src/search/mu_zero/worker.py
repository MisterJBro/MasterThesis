from torch.multiprocessing import Process
from src.search.mu_zero.core import MZCore


class MZWorker(Process, MZCore):
    """ Multiprocessing Tree Worker, for parallelization of MCTS."""
    def __init__(self, config, act_num, eval_channel, idx, channel):
        MZCore.__init__(self, config, None, act_num, eval_channel, idx=idx)
        Process.__init__(self)

        self.channel = channel

    def run(self):
        msg = self.channel.recv()
        while msg["command"] != "close":
            if msg["command"] == "search":
                self.set_root(msg["state"])
                qvals = self.search(msg["iters"])
                self.channel.send(qvals)
            msg = self.channel.recv()
