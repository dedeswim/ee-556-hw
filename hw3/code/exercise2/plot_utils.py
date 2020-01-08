import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

def plot_reconstruction(reconstruction, error_map, psnr, t, method):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # Plot image and error map
    ax[0].imshow(reconstruction, cmap="gray")
    ax[0].set_title(f'{method} - PSNR = {psnr}\n - Time: {t}s')
    im = ax[1].imshow(error_map, cmap="inferno", vmax=.05)
    
    # Plot the colorbar
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical');

    [axi.set_axis_off() for axi in ax.flatten()]

    plt.show()
