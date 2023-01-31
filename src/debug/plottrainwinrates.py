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
    #['default', 'classic', 'Solarize_Light2', 'bmh', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark', 'seaborn-dark-palette', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'tableau-colorblind10']
    with plt.style.context('seaborn-ticks'):
        # Values over several seeds
        y0 = np.array([
            [1.0, 0.995, 0.995, 1.0, 0.995, 1.0, 0.995, 0.995, 0.995, 1.0, 1.0, 1.0, 0.99, 1.0, 0.985, 1.0, 0.99, 0.995, 0.99, 0.995,
                0.98, 0.97, 1.0, 0.985, 0.975, 0.975, 0.965, 0.97, 0.99, 0.975, 0.985, 0.975, 0.98, 0.96, 0.96, 0.97, 0.965, 0.97, 0.98,
                0.98, 0.945, 0.965, 0.97, 0.945, 0.97, 0.985, 0.96, 0.945, 0.95, 0.95, 0.945, 0.975, 0.935, 0.935, 0.92, 0.96, 0.935, 0.92,
                0.965, 0.95, 0.925, 0.915, 0.935, 0.87, 0.855, 0.86, 0.875, 0.885, 0.86, 0.885, 0.855, 0.88, 0.845, 0.81, 0.745, 0.835,
                0.815, 0.825, 0.83, 0.74, 0.755, 0.695, 0.85, 0.7, 0.75, 0.75, 0.785, 0.71, 0.625, 0.655, 0.585, 0.635, 0.555, 0.5, 0.57,
                0.63, 0.58, 0.5, 0.515, 0.525]
        ])
        y1 = np.array([
            [0.53, 0.62, 0.55, 0.46, 0.49, 0.5, 0.57, 0.55, 0.64, 0.66, 0.61, 0.65, 0.64, 0.63, 0.59, 0.67, 0.62, 0.62, 0.49, 0.6, 0.54, 0.44, 0.61, 0.54, 0.37, 0.5, 0.59, 0.51, 0.67, 0.64, 0.56, 0.61, 0.6, 0.56, 0.45, 0.44, 0.47, 0.43, 0.4, 0.33, 0.44, 0.47, 0.31, 0.3, 0.29, 0.26, 0.33, 0.29, 0.24, 0.29, 0.32, 0.21, 0.24, 0.3, 0.29, 0.42, 0.47, 0.49, 0.34, 0.31, 0.42, 0.5, 0.49, 0.51, 0.5, 0.38, 0.53, 0.53, 0.56, 0.53, 0.41, 0.57, 0.5, 0.56, 0.59, 0.58, 0.55, 0.49, 0.48, 0.46, 0.42, 0.48, 0.37, 0.51, 0.56, 0.47, 0.42, 0.48, 0.63, 0.44, 0.48, 0.5, 0.5, 0.46, 0.44, 0.49, 0.54, 0.41, 0.49, 0.59, 0.68],
        ])
        y2 = np.array([
            [0.1, 0.2, 0.4, 0.5, 0.6, 0.4],
        ])
        y3 = np.array([
            [0.1, 0.2, 0.4, 0.5, 0.6, 0.4],
        ])


        # Statistics
        x = np.arange(y0.shape[-1])
        mean0 = np.mean(y0, 0)
        mean1 = np.mean(y1, 0)
        mean2 = np.mean(y2, 0)
        mean3 = np.mean(y3, 0)

        # Parameters
        fontsize_title = 18
        fontsize_label = 12
        xlim = (-1, 101)
        ylim = (-0.05, 1.05)
        xticks = np.arange(0, 110, step=10)
        yticks = np.arange(0, 1+0.5, step=0.5)
        grid = True
        title_pad = 10

        # Plot
        plt.figure(figsize=(15, 8), dpi=180)
        plt.subplot(211)
        plt.title("Naive SP", fontsize=fontsize_title,  fontweight="bold", pad=title_pad)
        plt.xlabel("Previous Versions", fontsize=fontsize_label, fontweight="bold")
        plt.ylabel("Win Rate", fontsize=fontsize_label, fontweight="bold")
        m, h1, h2 = mean_confidence_interval(y0)
        plt.plot(x, m, label="")
        plt.fill_between(x, h1, h2, alpha=0.4)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xticks(xticks, fontweight="bold")
        plt.yticks(yticks, fontweight="bold")
        plt.grid(grid)

        plt.subplot(212)
        plt.title("δ-Uniform SP", fontsize=fontsize_title,  fontweight="bold", pad=title_pad)
        plt.xlabel("Previous Versions", fontsize=fontsize_label, fontweight="bold")
        plt.ylabel("Win Rate", fontsize=fontsize_label, fontweight="bold")
        m, h1, h2 = mean_confidence_interval(y1)
        plt.plot(x, m, label="")
        plt.fill_between(x, h1, h2, alpha=0.4)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xticks(xticks, fontweight="bold")
        plt.yticks(yticks, fontweight="bold")
        plt.grid(grid)

        plt.subplots_adjust(left=0.044, bottom=0.06, right=0.996, top=0.95, wspace=0.138, hspace=0.45)
        #plt.tight_layout()
        plt.show()
