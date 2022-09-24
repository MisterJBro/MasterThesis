from multiprocessing import Process
from src.search.pgs.tree import PGSTree


class VEPGSTreeWorker(Process, PGSTree):
    """ Multiprocessing Tree Worker, for parallelization of MCTS."""
    def __init__(self, iters, eval_channel,  pol_head, val_head, idx, config, channel):
        Process.__init__(self)
        PGSTree.__init__(self, None, eval_channel, pol_head, val_head, config, idx=idx)

        self.iters = iters
        self.channel = channel

    def run(self):
        self.reset_policy()

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
            elif msg["command"] == "update":
                self.reset_policy(base_policy=msg["pol_head"], base_value=msg["val_head"])

            msg = self.channel.recv()