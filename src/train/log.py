import time
import numpy as np
from datetime import timedelta
import random
import numbers
import seaborn as sns
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
        sns.set_theme()
        if config["log_to_writer"]:
            self.writer = SummaryWriter(log_dir=config["experiment_path"], comment=f'{config["env"]}_{config["num_samples"]}')

    def __call__(self, metric, value, unit="", show=True):
        if metric in self:
            self[metric]["value"].append(value)
        else:
            self[metric] = {
                "value": [value],
                "unit": unit,
                "show": show
            }

    def __str__(self):
        current_time = time.time() - self.timer
        current_time = str(timedelta(seconds=int(current_time)))
        res = ''
        for name, data in self.items():
            if not data["show"]:
                continue
            val = data["value"][-1]
            unit = data["unit"]

            val_str = f'{val:01d}' if isinstance(val, int) else f'{val:.02f}'
            if unit == 's' or unit == '%':
                val_str = f'{val:.01f}'
            res += f'{str(name).capitalize()}: ' + val_str + unit  + '  '
        res += f'Time: {current_time}'
        return res

    def update(self, dict):
        for k, v in dict.items():
            self(k, v)

    def to_file(self, path=None):
        path = path if path is not None else self.log_path
        with open(path + '/' + self.config['log_name'], 'a') as f:
            f.write(str(self) + '\n')

    def to_writer(self):
        iter = self["iter"]["value"][-1]
        for name, data in self.items():
            val = data["value"][-1]
            if name == "games":
                # Prepare data
                data = {}
                opponents = self["sampled_policies"]["value"][-1]
                for k, v in val.items():
                    if k == (0,0):
                        data[-2] = [[v['win_base']/4, v['num']/2], [v['win_base']/4, v['num']/2]]
                    else:
                        m = max(k)
                        m = iter-1 if m==1 else opponents[m-2]
                        idx = 1 - np.nonzero(k)[0][0]
                        if m not in data:
                            data[m] = [[0, 0], [0, 0]]
                        data[m][idx] = [v['win_base'], v['num']]
                data = {k: v for k, v in sorted(data.items(), key=lambda item: -item[0])}
                fig, ax = plt.subplots(figsize=(6, len(data)), dpi=400)
                ax.invert_yaxis()
                colors = ["#c23119", "#dadb9d", "#238a3c"]
                x = np.arange(len(data))
                width = 0.3
                i = 0
                for k, v in data.items():
                    wr1 = np.round(v[0][0] / (v[0][1] + 1e-9), 2)
                    wr2 = np.round(v[1][0] / (v[1][1] + 1e-9), 2)

                    # Color based on winrate
                    rects1 = ax.barh(i - width/2, max(wr1, 0.15), height=width, color=colors[2] if wr1 > 0.55 else colors [0] if wr1 < 0.45 else colors[1])
                    rects2 = ax.barh(i + width/2, max(wr2, 0.15), height=width, color=colors[2] if wr2 > 0.55 else colors [0] if wr2 < 0.45 else colors[1])
                    ax.bar_label(rects1, labels=[f"{wr1} ({v[0][1]})"], label_type='center', color="white")
                    ax.bar_label(rects2, labels=[f"{wr2} ({v[1][1]})"], label_type='center', color="white")
                    i += 1
                ax.set_xlabel('Win Rate (%)')
                ax.set_title('Sampled games')
                ax.set_yticks(x, [f'Policy vs P{"olicy" if k==-2 else k}' for k in data.keys()])
                num_bars = len(data)
                plt.subplots_adjust(left=0.25, bottom=max(-0.0333*num_bars+0.2666, 0.1), right=0.9, top=min(0.025*num_bars + 0.8, 0.93))
                self.writer.add_figure(name, fig, iter)
            elif name == "elo":
                fig, ax = plt.subplots(dpi=400)
                ax.plot(data["value"])
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Elo')
                ax.set_title('Elo progression')
                self.writer.add_figure(name, fig, iter)
            elif isinstance(val, numbers.Number):
                self.writer.add_scalar(name, val, iter)

    def close(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

    def show(self):
        print(self)
        if self.config["log_to_file"]:
            self.to_file()
        if self.config["log_to_writer"]:
            self.to_writer()