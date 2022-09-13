import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    sns.set_theme()

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
    x = np.array([10, 20, 40, 60, 80, 100]).astype(str)
    mean0 = np.mean(y0, 0)
    #sigma0 = np.std(y0, 0)/3
    mean1 = np.mean(y1, 0)
    #sigma1 = np.std(y1, 0)/3
    mean2 = np.mean(y2, 0)
    #sigma2 = np.std(y2, 0)/3
    mean3 = np.mean(y3, 0)
    #sigma3 = np.std(y3, 0)/3

    # Plot
    #plt.plot(x, mean0, label="PGS long horizon")
    #plt.fill_between(x, mean0+sigma0, mean0-sigma0, alpha=0.4)
    plt.plot(x, mean1, label="MCS")
    #plt.fill_between(x, mean1+sigma1, mean1-sigma1, alpha=0.4)
    #plt.plot(x, mean2, label="PGS short horizon")
    #plt.fill_between(x, mean2+sigma2, mean2-sigma2, alpha=0.4)
    plt.plot(x, mean3, label="PGS long horizon")
    #plt.fill_between(x, mean3+sigma3, mean3-sigma3, alpha=0.4)

    plt.title('PGS Pdlm')
    plt.xlabel('Number of Iterations per Action')
    plt.ylabel('Return')
    plt.legend()
    plt.show()