import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    sns.set_theme()

    # Values over several seeds
    y1 = np.array([
        [64.731, 62.413, 60.762, 86.960, 86.169, 86.059, 87.646, 85.697, 84.240, 86.923, 86.578],
        [59.831, 83.569, 63.081, 86.771, 86.100, 86.859, 85.067, 86.654, 85.746, 86.428, 86.502],
        [59.675, 88.315, 86.659, 86.395, 86.592, 86.079, 87.412, 85.068, 86.974, 85.472, 85.715],
        [63.033, 60.824, 86.317, 85.643, 87.027, 86.755, 85.431, 85.516, 87.415, 87.807, 85.800],
    ])
    x1 = np.array([10, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000]).astype(str)


    # Statistics
    mean1 = np.mean(y1, 0)
    sigma1 = np.std(y1, 0)

    # Plot
    plt.plot(x1, mean1, label="MCS PUCT=20")
    plt.fill_between(x1, mean1+sigma1, mean1-sigma1, alpha=0.5)

    plt.title('MCS Pdlm')
    plt.xlabel('Number of Iterations per Action')
    plt.ylabel('Return')
    plt.legend()
    plt.show()