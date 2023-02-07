import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a, 0), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

if __name__ == '__main__':
    with plt.style.context('seaborn-ticks'):
        # Values over several seeds
        y0 = np.array([
            [0.9, 0.6, 0.7, 0.85],
            [0.85, 0.6, 0.65, 0.80],
            [0.8, 0.55, 0.65, 0.75],
        ])

        # Statistics
        x = np.array([25, 50, 100, 200])

        # Parameters
        fontsize_title = 16
        fontsize_label = 12
        xlim = (20, 205)
        ylim = (-0.05, 1.05)
        xticks = np.arange(50, 200+50, step=50)
        yticks = np.arange(0, 1+0.5, step=0.5)
        grid = True
        title_pad = 10

        # Plot
        plt.figure(figsize=(8, 5), dpi=180)
        plt.title("Combinations of Extensions", fontsize=fontsize_title,  fontweight="bold", pad=title_pad)
        plt.xlabel("Previous Versions", fontsize=fontsize_label, fontweight="bold")
        plt.ylabel("Win Rate", fontsize=fontsize_label, fontweight="bold")
        plt.axhline(y = 0.5, color='orange', linestyle = '--')
        m, h1, h2 = mean_confidence_interval(y0)
        plt.plot(x, m, label="")
        plt.fill_between(x, h1, h2, alpha=0.4)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xticks(xticks, fontweight="bold")
        plt.yticks(yticks, fontweight="bold")
        plt.legend(loc="upper left", fontsize=12)
        plt.grid(grid)

        plt.show()
