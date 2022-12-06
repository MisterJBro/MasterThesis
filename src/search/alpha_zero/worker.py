from torch.multiprocessing import Process
from src.search.alpha_zero.core import AZCore


class AZWorker(Process, AZCore):
    """ Multiprocessing Tree Worker, for parallelization of MCTS."""
    def __init__(self, config, eval_channel, idx, channel):
        AZCore.__init__(self, config, None, eval_channel, idx=idx)
        Process.__init__(self)

        self.channel = channel

    def run(self):
        msg = self.channel.recv()
        while msg["command"] != "close":
            if msg["command"] == "search":
                self.set_root(msg["state"])
                res = self.search(msg["iters"])
                self.channel.send(res)
            msg = self.channel.recv()
