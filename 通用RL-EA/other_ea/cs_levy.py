import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def step_data(data, m):
    return [np.mean(data[i:i + m]) for i in range(0, len(data), m)]

def plot_step_histogram_with_kde(data, m):
    stepped_data = step_data(data, m)
    plt.bar(range(len(stepped_data)), stepped_data, width=1.0, align='edge', alpha=0.6, label='Step Histogram')
    sns.kdeplot(data, bw_adjust=0.5, fill=True, label='KDE')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.title('Step Histogram with KDE')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    data2 = [40, 37, 30, 33, 35, 32, 29, 31, 25, 20]
    m = 2  # Change this value to adjust the step size
    plot_step_histogram_with_kde(data2, m)
