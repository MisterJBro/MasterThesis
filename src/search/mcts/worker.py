from torch.multiprocessing import Process
from src.search.mcts.core import MCTSCore


class MCTSWorker(Process, MCTSCore):
    """ Sinlge MCTS Worker for parallelization of MCTS."""
    def __init__(self, config, channel):
        MCTSCore.__init__(self, None, config)
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
