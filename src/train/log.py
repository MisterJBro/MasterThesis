import time
import numpy as np
from datetime import timedelta

from torch.utils.tensorboard import SummaryWriter

class Logger(dict):
    """Basic logger for metrics"""

    def __init__(self, config, path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.log_path = path
        self.main_metric = []
        self.timer = time.time()
        self.writer = None
        if config["log_to_writer"]:
            self.writer = SummaryWriter(log_dir="../runs",comment=f'{config["env"]}_{config["num_samples"]}')

    def __call__(self, metric, value):
        if metric == self.config["log_main_metric"]:
            self.main_metric.append(value)

        self[metric] = value

    def __str__(self):
        current_time = time.time() - self.timer
        current_time = str(timedelta(seconds=int(current_time)))
        return '  '.join([f'{str(k).capitalize()}: ' + (f'{v:01d}' if isinstance(v, int) else f'{v:.02f}') for k, v in self.items()]) + f'  Time: {current_time}'

    def update(self, dict):
        for k, v in dict.items():
            self(k, v)

    def to_file(self, path=None):
        path = path if path is not None else self.log_path
        with open(path + '/' + self.config['log_name'], 'a') as f:
            f.write(str(self) + '\n')

    def to_writer(self, iter):
        for k, v in self.items():
            self.writer.add_scalar(str(k), v, iter)

    def close(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()