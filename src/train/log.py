import time
import numpy as np

class Logger(dict):
    """Basic logger for metrics"""

    def __init__(self, config, path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.log_path = path
        self.best_model_path = ""
        self.best_metric = float('-inf')
        self.timer = time.time()

    def __call__(self, metric, value):
        self[metric] = value

    def __str__(self):
        current_time = time.time() - self.timer
        return 'Time: {current_time:.02f}  ' + '  '.join([f'{str(k).capitalize()}:' + f'{v:.02f}' for k, v in self.items()])

    def update(self, dict):
        for k, v in dict.items():
            self(k, v)

    def to_file(self, path=None):
        path = path if path is not None else self.log_path
        with open(path + '/' + self.config['log_name'], 'a') as f:
            f.write(str(self) + '\n')
