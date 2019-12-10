import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np


def plot_convergence(results, fs_star, title, ylabel):
    colors = {'ISTA': 'red', 'FISTA': 'blue', 'FISTA-RESTART': 'green'}
    ax1 = plt.subplot(1, 1, 1)
    for key in ['ISTA', 'FISTA', 'FISTA-RESTART']:
        if key in results:
            num_iterations = results[key]['iters'] + 1
            ax1.semilogy(np.array(range(0, num_iterations)), abs(results[key]['conv'][:num_iterations] - fs_star) / fs_star,
                     color=colors[key], lw=2, label=key)
    ax1.legend()
    ax1.set_xlabel('#iterations')
    ax1.set_ylabel(ylabel)
    ax1.grid()
    plt.suptitle(title)
    plt.show()
