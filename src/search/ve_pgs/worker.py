from multiprocessing import Process
from src.search.ve_pgs.core import VEPGSCore


class VEPGSWorker(Process, VEPGSCore):
    """ Multiprocessing Tree Worker, for parallelization of MCTS."""
    def __init__(self, config, num_acts, eval_channel, pol_head, val_head, idx, channel):
        Process.__init__(self)
        VEPGSCore.__init__(self, config, None, num_acts, eval_channel, pol_head, val_head, idx=idx)

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