import time
import numpy as np
from datetime import timedelta
import random

from torch.utils.tensorboard import SummaryWriter

class Logger(dict):
    """Basic logger for metrics"""

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.log_path = config['log_path']
        self.main_metric = []
        self.timer = time.time()
        self.writer = None
        if config["log_to_writer"]:
            self.writer = SummaryWriter(log_dir="../runs",comment=f'{config["env"]}_{config["num_samples"]}')

    def __call__(self, metric, value, unit="", show=True):
        if metric == self.config["log_main_metric"]:
            self.main_metric.append(value)

        self[metric] = (value, unit, show)

    def __str__(self):
        current_time = time.time() - self.timer
        current_time = str(timedelta(seconds=int(current_time)))
        res = ''
        for name, (val, unit_str, show) in reversed(self.items()):
            if not show:
                continue

            val_str = f'{val:01d}' if isinstance(val, int) else f'{val:.02f}'
            if unit_str == 's' or unit_str == '%':
                val_str = f'{val:.01f}'
            res += f'{str(name).capitalize()}: ' + val_str + unit_str  + '  '
        res += f'Time: {current_time}'
        return res

    def update(self, dict):
        for k, v in dict.items():
            self(k, v)

    def to_file(self, path=None):
        path = path if path is not None else self.log_path
        with open(path + '/' + self.config['log_name'], 'a') as f:
            f.write(str(self) + '\n')

    def to_writer(self, iter):
        for name, (val, unit_str, show) in self.items():
            self.writer.add_scalar(str(name), val, iter)

    def close(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()