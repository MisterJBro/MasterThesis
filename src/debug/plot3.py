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
        dyn_len = np.array([
            [0.725, 0.8, 0.675, 0.175],
            [0.675, 0.65, 0.675, 0.45],
            [0.675, 0.575, 0.45, 0.45],
            [0.675, 0.575, 0.525, 0.4],
            [0.675, 0.775, 0.55, 0.375],
        ])
        scale_vals = np.array([
            [0.75, 0.4, 0.5, 0.65],
            [0.575, 0.75, 0.475, 0.65],
            [0.675, 0.675, 0.525, 0.525],
            [0.75, 0.625, 0.475, 0.575],
            [0.7, 0.675, 0.5, 0.625],
        ])
        expl_entr = np.array([
            [0.525, 0.475, 0.4, 0.45],
            [0.5, 0.425, 0.425, 0.425],
            [0.575, 0.375, 0.5, 0.425],
            [0.425, 0.525, 0.35, 0.3],
            [0.45, 0.55, 0.575, 0.4],
        ])
        expl_kl = np.array([
            [0.55, 0.5, 0.525, 0.625],
            [0.475, 0.525, 0.475, 0.4],
            [0.625, 0.4, 0.5, 0.475],
            [0.525, 0.4, 0.375, 0.475],
            [0.525, 0.45, 0.575, 0.575],
        ])
        visit_counts = np.array([
            [0.125, 0.075, 0.125, 0.175],
            [0.05, 0.15, 0.225, 0.125],
            [0.1, 0.15, 0.075, 0.225],
            [0.1, 0.075, 0.175, 0.175],
            [0.2, 0.175, 0.2, 0.15],
        ])
        pgs_update = np.array([
            [0.525, 0.375, 0.4, 0.25],
            [0.5, 0.45, 0.275, 0.225],
            [0.475, 0.35, 0.35, 0.25],
            [0.575, 0.375, 0.35, 0.15],
            [0.45, 0.3, 0.3, 0.325],
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
        plt.figure(figsize=(9, 4), dpi=180)
        plt.subplot(231)
        plt.title("Dynamic Length", fontsize=fontsize_title,  fontweight="bold", pad=title_pad)
        #plt.xlabel("Iterations", fontsize=fontsize_label, fontweight="bold")
        plt.ylabel("Win Rate", fontsize=fontsize_label, fontweight="bold")
        m, h1, h2 = mean_confidence_interval(dyn_len)
        plt.plot(x, m)
        plt.fill_between(x, h1, h2, alpha=0.4)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xticks(xticks, fontweight="bold")
        plt.yticks(yticks, fontweight="bold")
        plt.grid(grid)
        #plt.fill_between(x, mean0+sigma0, mean0-sigma0, alpha=0.4)

        plt.subplot(232)
        plt.title("Weight Values", fontsize=fontsize_title,  fontweight="bold", pad=title_pad)
        #plt.xlabel("Iterations", fontsize=fontsize_label, fontweight="bold")
        #plt.ylabel("Win Rate", fontsize=fontsize_label, fontweight="bold")
        m, h1, h2 = mean_confidence_interval(scale_vals)
        plt.plot(x, m)
        plt.fill_between(x, h1, h2, alpha=0.4)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xticks(xticks, fontweight="bold")
        plt.yticks(yticks, fontweight="bold")
        plt.grid(grid)
        #plt.fill_between(x, mean1+sigma1, mean1-sigma1, alpha=0.4)

        plt.subplot(233)
        plt.title("Entropy Bonus", fontsize=fontsize_title,  fontweight="bold", pad=title_pad)
        #plt.xlabel("Iterations", fontsize=fontsize_label, fontweight="bold")
        #plt.ylabel("Win Rate", fontsize=fontsize_label, fontweight="bold")
        m, h1, h2 = mean_confidence_interval(expl_entr)
        plt.plot(x, m)
        plt.fill_between(x, h1, h2, alpha=0.4)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xticks(xticks, fontweight="bold")
        plt.yticks(yticks, fontweight="bold")
        plt.grid(grid)
        #plt.fill_between(x, mean2+sigma2, mean2-sigma2, alpha=0.4)

        plt.subplot(234)
        plt.title("KL Divergence", fontsize=fontsize_title,  fontweight="bold", pad=title_pad)
        plt.xlabel("Iterations", fontsize=fontsize_label, fontweight="bold")
        plt.ylabel("Win Rate", fontsize=fontsize_label, fontweight="bold")
        m, h1, h2 = mean_confidence_interval(expl_kl)
        plt.plot(x, m)
        plt.fill_between(x, h1, h2, alpha=0.4)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xticks(xticks, fontweight="bold")
        plt.yticks(yticks, fontweight="bold")
        plt.grid(grid)
        #plt.fill_between(x, mean2+sigma2, mean2-sigma2, alpha=0.4)

        plt.subplot(235)
        plt.title("Policy Target", fontsize=fontsize_title,  fontweight="bold", pad=title_pad)
        plt.xlabel("Iterations", fontsize=fontsize_label, fontweight="bold")
        #plt.ylabel("Win Rate", fontsize=fontsize_label, fontweight="bold")
        m, h1, h2 = mean_confidence_interval(visit_counts)
        plt.plot(x, m)
        plt.fill_between(x, h1, h2, alpha=0.4)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xticks(xticks, fontweight="bold")
        plt.yticks(yticks, fontweight="bold")
        plt.grid(grid)
        #plt.fill_between(x, mean2+sigma2, mean2-sigma2, alpha=0.4)

        plt.subplot(236)
        plt.title("Update", fontsize=fontsize_title,  fontweight="bold", pad=title_pad)
        plt.xlabel("Iterations", fontsize=fontsize_label, fontweight="bold")
        #plt.ylabel("Win Rate", fontsize=fontsize_label, fontweight="bold")
        m, h1, h2 = mean_confidence_interval(pgs_update)
        plt.plot(x, m)
        plt.fill_between(x, h1, h2, alpha=0.4)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xticks(xticks, fontweight="bold")
        plt.yticks(yticks, fontweight="bold")
        plt.grid(grid)
        #plt.fill_between(x, mean2+sigma2, mean2-sigma2, alpha=0.4)

        plt.subplots_adjust(left=0.075, bottom=0.12, right=0.99, top=0.92, wspace=0.36, hspace=0.8)
        #plt.tight_layout()
        plt.show()
