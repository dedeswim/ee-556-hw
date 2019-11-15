import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np

def plot_convergence(results, fs_star, epoch_to_iteration_exchange_rate, title):
    colors = { 'ISTA': 'red', 'FISTA': 'blue', 'FISTA-RESTART': 'green', 'PROX-SG': 'magenta'}
    # DETERM
    ax1 = plt.subplot(1, 2, 1)
    for key in ['ISTA', 'FISTA', 'FISTA-RESTART']:
        if key in results:
            num_iterations = len(results[key]['conv'])
            ax1.plot(np.array(range(0, num_iterations)), abs(results[key]['conv'] - fs_star) / fs_star,
                                                                color=colors[key], lw=2, label=key)
    ax1.legend()
    ax1.set_xlabel('#iterations')
    ax1.set_ylabel(r'$ |f(\mathbf{x}^k) - f^\star|  /  f^\star$')
    ax1.set_ylim(1e-7, 1e2)
    ax1.set_yscale('log')
    ax1.grid()

    # STOCHASTIC
    ax2 = plt.subplot(1, 2, 2)
    ticks = []
    locs = []
    for key in ['PROX-SG', 'PROX-SVRG++']:
        if key in results:
            num_iterations = len(results[key]['conv'])
            ax2.plot(np.array(range(0, num_iterations)), abs(results[key]['conv'] - fs_star) / fs_star,
                                                                color=colors[key], lw=2, label=key)
            num_epochs = num_iterations / epoch_to_iteration_exchange_rate
            tick_frec = 50
            tcks = [tick_frec*i for i in range(1, int(num_epochs/tick_frec) + 1)]
            locs = [epoch_to_iteration_exchange_rate*tick_frec*i for i in range(1, int(num_epochs/tick_frec) + 1)]
            plt.xticks(locs, tcks)
    ax2.legend()
    ax2.set_xlabel('#epochs')
    ax2.set_ylabel(r'$ |f(\mathbf{x}^k) - f^\star|  /  f^\star$')
    ax2.set_ylim(1e-2, 1e2)
    ax2.set_yscale('log')
    ax2.set_xticks(locs,ticks)
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(8)
    ax2.grid()
    plt.suptitle(title)
    plt.show()



def plot_digit_features(X, title):
    plt.figure(figsize=(10, 5))
    scale = np.abs(X).max()
    for i in range(10):
        l1_plot = plt.subplot(2, 5, i + 1)
        l1_plot.imshow(X[:, i].reshape(28, 28), interpolation='nearest',
                       cmap=plt.cm.RdBu, vmin=-scale, vmax=scale)
        l1_plot.set_xticks(())
        l1_plot.set_yticks(())
        l1_plot.set_xlabel('Class %i' % i)
    plt.suptitle(title)
    plt.show()
