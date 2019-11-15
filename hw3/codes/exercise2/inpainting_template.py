import time
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle
from skimage.measure import compare_ssim as ssim

from utils import apply_random_mask, psnr, load_image
from operators import TV_norm, RepresentationOperator, p_omega, p_omega_t, l1_prox



def ISTA(fx, gx, gradf, proxg, params):
## TO BE FILLED ##


def FISTA(fx, gx, gradf, proxg, params, verbose=False):
## TO BE FILLED ##

def reconstruct_l1(image, indices, optimizer, params):
    # Wavelet operator
    r = Representation_Operator(m=params["m"])

    # Define the overall operator
    forward_operator = lambda x: ## TO BE FILLED ##  # P_Omega.W^T
    adjoint_operator = lambda x: ## TO BE FILLED ##  # W. P_Omega^T

    # Generate measurements
    b = ## TO BE FILLED ##

    fx = lambda x: ## TO BE FILLED ##
    gx = lambda x: ## TO BE FILLED ##
    proxg = lambda x, y: l1_prox(x, params['lambda'] * y)
    gradf = lambda x: ## TO BE FILLED ##

    x, info = optimizer(fx, gx, gradf, proxg, params)
    return r.WT(x).reshape((params['m'], params['m']), order="F"), info


def reconstruct_TV(image, indices, optimizer, params):
    """
        image: undersampled image (mxm) to be reconstructed
        indices: indices of the undersampled locations
        optimizer: method of reconstruction (FISTA/ISTA function handle)
        params:
    """
    # Define the overall operator
    forward_operator = lambda x: ## TO BE FILLED ##  # P_Omega.W^T
    adjoint_operator = lambda x: ## TO BE FILLED ##  # W. P_Omega^T

    # Generate measurements
    b = ## TO BE FILLED ##

    fx = lambda x: ## TO BE FILLED ##
    gx = lambda x: ## TO BE FILLED ##
    proxg = lambda x, y: denoise_tv_chambolle(x.reshape((params['m'], params['m']), order="F"),
                                              weight=params["lambda"] * y, eps=1e-5,
                                              n_iter_max=50).reshape((params['N'], 1), order="F")
    gradf = lambda x: ## TO BE FILLED ##

    x, info = optimizer(fx, gx, gradf, proxg, params)
    return x.reshape((params['m'], params['m']), order="F"), info


# %%

if __name__ == "__main__":

    ##############################
    # Load image and sample mask #
    ##############################
    shape = (256, 256)
    params = {
        'maxit': 200,
        'tol': 10e-15,
        'Lips': ## TO BE FILLED ##,
        'lambda': ## TO BE FILLED ##,
        'x0': np.zeros((shape[0] * shape[1], 1)),
        'restart_criterion': ## TO BE FILLED ##, gradient_scheme,
        'stopping_criterion': ## TO BE FILLED ##,
        'iter_print': 50,
        'shape': shape,
        'restart_param': 50,
        'verbose': True,
        'm': shape[0],
        'rate': 0.4,
        'N': shape[0] * shape[1]
    }
    PATH = ## TO BE FILLED ##
    image = load_image(PATH, params['shape'])

    im_us, mask = apply_random_mask(image, params['rate'])
    indices = np.nonzero(mask.flatten(order='F'))[0]
    params['indices'] = indices
    # Choose optimization parameters


    #######################################
    # Reconstruction with L1 and TV norms #
    #######################################
    t_start = time.time()
    reconstruction_l1 = reconstruct_l1(image, indices, FISTA, params)[0]
    t_l1 = time.time() - t_start

    psnr_l1 = psnr(image, reconstruction_l1)
    ssim_l1 = ssim(image, reconstruction_l1)

    t_start = time.time()
    reconstruction_tv = reconstruct_TV(image, indices, FISTA, params)[0]
    t_tv = time.time() - t_start

    psnr_tv = psnr(image, reconstruction_tv)
    ssim_tv = ssim(image, reconstruction_tv)

    # Plot the reconstructed image alongside the original image and PSNR
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original')
    ax[1].imshow(im_us, cmap='gray')
    ax[1].set_title('Original with missing pixels')
    ax[2].imshow(reconstruction_l1, cmap="gray")
    ax[2].set_title('L1 - PSNR = {:.2f}\n SSIM  = {:.2f} - Time: {:.2f}s'.format(psnr_l1, ssim_l1, t_l1))
    ax[3].imshow(reconstruction_tv, cmap="gray")
    ax[3].set_title('TV - PSNR = {:.2f}\n SSIM  = {:.2f}  - Time: {:.2f}s'.format(psnr_tv, ssim_tv, t_tv))
    [axi.set_axis_off() for axi in ax.flatten()]
    plt.tight_layout()
    plt.show()
