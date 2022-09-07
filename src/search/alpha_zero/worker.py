from multiprocessing import Process
from src.search.alpha_zero.tree import AZTree


class AZTreeWorker(Process, AZTree):
    """ Multiprocessing Tree Worker, for parallelization of MCTS."""
    def __init__(self, iters, eval_channel, idx, config, channel):
        AZTree.__init__(self, None, eval_channel, config, idx=idx)
        Process.__init__(self)

        self.iters = iters
        self.channel = channel

    def run(self):
        msg = self.channel.recv()
        while msg["command"] != "close":
            if msg["command"] == "search":
                self.set_root(msg["state"])
                if msg["iters"] is not None:
                    iters = msg["iters"]
                else:
                    iters = self.iters
                qvals = self.search(iters)
                self.channel.send(qvals)
            msg = self.channel.recv()
