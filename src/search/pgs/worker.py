from multiprocessing import Process
from src.search.pgs.core import PGSCore


class PGSWorker(Process, PGSCore):
    """ Worker Process for PGS."""

    def __init__(self, config, eval_channel, pol_head, val_head, idx, channel):
        Process.__init__(self)
        PGSCore.__init__(self, config, None, eval_channel, pol_head, val_head, idx=idx)

        self.channel = channel

    def run(self):
        self.reset()
        msg = self.channel.recv()
        while msg["command"] != "close":
            if msg["command"] == "search":
                self.reset()
                self.set_root(msg["state"])
                qvals = self.search(msg["iters"])
                self.channel.send(qvals)
            elif msg["command"] == "update":
                self.reset(base_policy=msg["pol_head"], base_value=msg["val_head"])

            msg = self.channel.recv()