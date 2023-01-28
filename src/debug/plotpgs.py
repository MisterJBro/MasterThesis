import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    #['default', 'classic', 'Solarize_Light2', 'bmh', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark', 'seaborn-dark-palette', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'tableau-colorblind10']
    with plt.style.context('seaborn-ticks'):
        # Values over several seeds
        y0 = np.array([
            [66.6, 66.88, 69.51, 71.57, 68.1, 68.55],
        ])
        y1 = np.array([
            [66.46, 67.52, 63.24, 68.22, 67.99, 67.12],
        ])
        y2 = np.array([
            [66.57, 67.9, 69.5, 68.06, 67.88, 68.43],
        ])
        y3 = np.array([
            [45.529, 25.494, 39.361, 66.441, 48.360, 55],
        ])


        # Statistics
        x = np.array([10, 20, 40, 60, 80, 100])
        mean0 = np.mean(y0, 0)
        #sigma0 = np.std(y0, 0)/3
        mean1 = np.mean(y1, 0)
        #sigma1 = np.std(y1, 0)/3
        mean2 = np.mean(y2, 0)
        #sigma2 = np.std(y2, 0)/3
        mean3 = np.mean(y3, 0)
        #sigma3 = np.std(y3, 0)/3

        # Parameters
        fontsize_title = 18
        fontsize_label = 13
        xlim = (-5, 205)
        ylim = (-0.1, 1.1)
        xticks = np.arange(0, 200, step=50)
        yticks = np.arange(0, 1, step=0.2)

        # Plot
        plt.figure(figsize=(15, 3), dpi=180)
        plt.subplot(141)
        plt.title("Variance Extension", fontsize=fontsize_title)
        plt.xlabel("Iterations", fontsize=fontsize_label, fontweight="bold")
        plt.ylabel("Win Rate", fontsize=fontsize_label, fontweight="bold")
        plt.plot(x, mean0, label="PGS long horizon")
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xticks(xticks)
        plt.yticks(yticks)
        #plt.fill_between(x, mean0+sigma0, mean0-sigma0, alpha=0.4)

        plt.subplot(142)
        plt.title("Exploration Extension", fontsize=fontsize_title)
        plt.xlabel("Iterations", fontsize=fontsize_label, fontweight="bold")
        plt.ylabel("Win Rate", fontsize=fontsize_label, fontweight="bold")
        plt.plot(x, mean1, label="PGS short horizon")
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xticks(xticks)
        plt.yticks(yticks)
        #plt.fill_between(x, mean1+sigma1, mean1-sigma1, alpha=0.4)

        plt.subplot(143)
        plt.title("Visit Counts Extension", fontsize=fontsize_title)
        plt.xlabel("Iterations", fontsize=fontsize_label, fontweight="bold")
        plt.ylabel("Win Rate", fontsize=fontsize_label, fontweight="bold")
        plt.plot(x, mean2, label="PGS short horizon")
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xticks(xticks)
        plt.yticks(yticks)
        #plt.fill_between(x, mean2+sigma2, mean2-sigma2, alpha=0.4)

        plt.subplot(144)
        plt.title("Update Extension", fontsize=fontsize_title)
        plt.xlabel("Iterations", fontsize=fontsize_label, fontweight="bold")
        plt.ylabel("Win Rate", fontsize=fontsize_label, fontweight="bold")
        plt.plot(x, mean3, label="PGS long horizon")
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xticks(xticks)
        plt.yticks(yticks)
        #plt.fill_between(x, mean3+sigma3, mean3-sigma3, alpha=0.4)

        plt.subplots_adjust(left=0.033, bottom=0.13, right=0.99, top=0.9, wspace=0.33)
        #plt.tight_layout()
        plt.show()
