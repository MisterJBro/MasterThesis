class Logger(dict):
    """Basic logger for metrics"""

    def __init__(self, path=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_path = path
        self.best_model_path = ""
        self.best_metric = float('-inf')

    def __call__(self, metric, value):
        self[metric] = value

    def __format__(self):
        return '  '.join(f'{str(k).capitalize()}: {f"{v:.2f}" if isinstance(v, float) else v}' for k, v in self.items())

    def update(self, dict):
        for k, v in dict.items():
            self(k, v)
