import sys
sys.path.append('../')

import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage.restoration import denoise_tv_chambolle
from skimage.measure import compare_ssim as ssim
from plot_utils import plot_convergence, plot_reconstruction
from utils import print_end_message, print_start_message, print_progress, apply_random_mask, psnr, load_image
from operators import TV_norm, RepresentationOperator, p_omega, p_omega_t, l1_prox
from unrolled_network import ResNetDC


def ISTA(fx, gx, gradf, proxg, params):
    method_name = 'ISTA'
    print_start_message(method_name)

    tic = time.time()

    # Get parameters
    lmbd = params['lambda']
    maxit = params['maxit']
    x = params['x0']
    alpha = 1 / params['Lips']

    run_details = {'X_final': None, 'conv': np.zeros(params['maxit'] + 1)}
    run_details['conv'][0] = fx(params['x0']) + gx(params['x0'])

    for k in range(maxit):

        # Compute prox
        x_next = proxg(x - alpha * gradf(x), alpha)

        # Record convergence
        run_details['conv'][k] = fx(x) + gx(x)

        if k % params['iter_print'] == 0:
            print_progress(k, params['maxit'],
                           run_details['conv'][k], fx(x), gx(x))

        # Get stopping criterion and tolerance
        stopping_criterion = params['stopping_criterion']
        tol = params['tol']

        # If error criterion (ex. 2.3.b) is required check
        if params['f_star_criterion']:
            f_star = params['f_star']
            f_k = run_details['conv'][k]

            # Stop if criterion is satisfied
            if stopping_criterion(f_k, f_star) < tol:
                break

        x = x_next

    run_details['conv'][k + 1] = fx(x) + gx(x)
    run_details['iters'] = k + 1

    run_details['X_final'] = x

    print_end_message(method_name, time.time() - tic)

    return x, run_details


def FISTA(fx, gx, gradf, proxg, params, verbose=False):
    if params['restart_fista']:
        method_name = 'FISTA-RESTART'
    else:
        method_name = 'FISTA'
    print_start_message(method_name)

    tic = time.time()

    # Get parameters
    stopping_criterion = params['stopping_criterion']
    tol = params['tol']
    lmbd = params['lambda']
    maxit = params['maxit']
    alpha = 1 / params['Lips']
    x = params['x0']
    y = x
    t = 1

    run_details = {'X_final': None, 'conv': np.zeros(params['maxit'] + 1)}
    run_details['conv'][0] = fx(params['x0']) + gx(params['x0'])

    for k in range(maxit):

        # Compute proximal
        x_next = proxg(y - alpha * gradf(y), alpha)

        # Compute momentum
        t_next = (1 + np.sqrt(4 * (t ** 2) + 1)) / 2
        y_next = x_next + (t - 1) / t_next * (x_next - x)

        # Check if FISTA Restart and if restart condition is satisfied
        if params['restart_fista'] and gradient_scheme_restart_condition(x, x_next, y):
            print('Restarting...')

            # If both are True, reset momentum and re-compute prox
            t_next = 1
            y_next = x_next
            x_next = proxg(y_next - alpha * gradf(y_next), alpha)

        # Record convergence
        run_details['conv'][k] = fx(x) + gx(x)

        if k % params['iter_print'] == 0:
            print_progress(k, params['maxit'],
                           run_details['conv'][k], fx(x_next), gx(x_next))

        # If error criterion (ex. 2.3.b) is required check
        if params['err_criterion']:
            if k == 0:
                # Save first error for next loops
                err_0 = np.linalg.norm(y_next - y)
            else:
                # Check if criterion is satisfied
                err_k = np.linalg.norm(y_next - y)
                if stopping_criterion(err_k, err_0) < tol:
                    break

        # If f* criterion (ex. 2.3.b) is required check
        if params['f_star_criterion']:
            f_star = params['f_star']
            f_k = run_details['conv'][k]
            # Check if criterion is satisfied
            if stopping_criterion(f_k, f_star) < tol:
                break

        x = x_next
        y = y_next
        t = t_next

    run_details['conv'][k + 1] = fx(x) + gx(x)
    run_details['iters'] = k + 1

    run_details['X_final'] = x

    print_end_message(method_name, time.time() - tic)

    return x, run_details


def gradient_scheme_restart_condition(X_k, X_k_next, Y_k):
    return np.trace((Y_k - X_k_next).T @ (X_k_next - X_k)) > 0


def reconstruct_l1(image, indices, optimizer, params):
    m = params['m']

    # Wavelet operator
    r = RepresentationOperator(m=m)

    # Define the overall operator
    def forward_operator(x): return p_omega(r.WT(x), indices)  # P_Omega.W^T
    def adjoint_operator(x): return r.W(
        p_omega_t(x, indices, m))  # W. P_Omega^T

    # Generate measurements, the image is vectorized
    b = p_omega(image.reshape(-1), indices)

    # Get parameters
    lmbd = params['lambda']

    def fx(x): return 1 / 2 * np.linalg.norm(b - forward_operator(x)) ** 2
    def gx(x): return lmbd * np.linalg.norm(x, 1)
    def proxg(x, y): return l1_prox(x, lmbd * y)
    def gradf(x): return - (adjoint_operator(b -
                                             forward_operator(x))).reshape(-1, 1)

    x, info = optimizer(fx, gx, gradf, proxg, params)

    return r.WT(x).reshape((params['m'], params['m'])), info


def reconstruct_TV(image, indices, optimizer, params):
    """
        image: undersampled image (mxm) to be reconstructed
        indices: indices of the undersampled locations
        optimizer: method of reconstruction (FISTA/ISTA function handle)
        params:
    """
    m = params["m"]

    r = RepresentationOperator(m=m)

    # Define the overall operator
    def forward_operator(x): return p_omega(r.WT(x), indices)  # P_Omega.W^T
    def adjoint_operator(x): return r.W(
        p_omega_t(image, indices, m))  # W. P_Omega^T

    # Generate measurements, the image is vectorized
    b = p_omega(image.reshape(-1), indices)

    def fx(x): return 1 / 2 * np.linalg.norm(b - p_omega(x, indices)) ** 2
    def gx(x): return params["lambda"] * TV_norm(x)

    def proxg(x, y): return denoise_tv_chambolle(x.reshape((params['m'], m)),
                                                 weight=params["lambda"] * y, eps=1e-5,
                                                 n_iter_max=50).reshape((params['N'], 1))

    def gradf(x): return - (p_omega_t(
        b - p_omega(x, indices), indices, m)).reshape(-1, 1)

    x, info = optimizer(fx, gx, gradf, proxg, params)

    return x.reshape((m, m)), info


def call_all_methods(image, indices, params, reconstruct):
    all_results = dict()

    all_results['ISTA'] = reconstruct(image, indices, ISTA, params)[1]

    params['restart_fista'] = False
    all_results['FISTA'] = reconstruct(image, indices, FISTA, params)[1]

    params['restart_fista'] = True
    all_results['FISTA-RESTART'] = reconstruct(
        image, indices, FISTA, params)[1]

    print(all_results['FISTA-RESTART']['iters'])

    return all_results

def compare_unrolled(params, image, indices):
    # Run FISTA the params defined above

    # Reconstruct using l-1 norm
    t_start = time.time()
    reconstruction_l1 = reconstruct_l1(image, indices, FISTA, params)[0]
    t_l1 = time.time() - t_start
    # Compute PSNR
    psnr_l1 = psnr(image, reconstruction_l1)
    # Compute error map
    error_map_l1 = abs(image - reconstruction_l1)
    # Plot reconstruction
    plot_reconstruction(reconstruction_l1, error_map_l1, psnr_l1, t_l1, 'L-1')

    # Reconstruct using TV norm
    t_start = time.time()
    reconstruction_TV = reconstruct_TV(image, indices, FISTA, params)[0]
    t_TV = time.time() - t_start
    # Compute PSNR
    psnr_TV = psnr(image, reconstruction_TV)
    # Compute error map
    error_map_TV = abs(image - reconstruction_TV)
    # Plot reconstruction
    plot_reconstruction(reconstruction_TV, error_map_TV, psnr_TV, t_TV, 'TV')

    # Initialize NN with 500 steps
    model = ResNetDC(2, unroll_depth=params['maxit'])
    model.load_state_dict(torch.load('data/unrolled_nn.pt'))

    # Take all the arguments to PyTorch tensors
    image_torch = torch.tensor(image).view(1, 1, params['shape'][0], params['shape'][1]).float()
    mask_torch = torch.tensor(mask).view(1, 1, params['shape'][0], params['shape'][1]).float()
    im_us_torch = image_torch * mask_torch

    # Reconstruct with NN
    with torch.no_grad():
        # Reconstruct using unrolling
        print_start_message('Unrolled NN')
        t_start = time.time()
        reconstruction_nn = model(im_us_torch, mask_torch)
        t_nn = time.time() - t_start
        print_end_message('Unrolled NN', t_nn)
        # Go back to NumPy array
        reconstruction_nn = reconstruction_nn[0, 0, :, :].cpu().numpy()
        # Compute PSNR
        psnr_nn = psnr(image, reconstruction_nn)
        # Compute error map
        error_map_nn = abs(image - reconstruction_nn)
        # Plot reconsruction
        plot_reconstruction(reconstruction_nn, error_map_nn, psnr_nn, t_nn, 'NN')


if __name__ == "__main__":

    ##############################
    # Load image and sample mask #
    ##############################
    r = RepresentationOperator()
    shape = (256, 256)

    # Set initial params
    params = {
        'maxit': 200,
        'tol': 1e-15,
        'Lips': 1,
        'lambda': 0.01,
        'x0': np.zeros((shape[0] * shape[1], 1)),
        'restart_criterion':  gradient_scheme_restart_condition,
        'iter_print': 50,
        'shape': shape,
        'restart_param': 50,
        'verbose': True,
        'm': shape[0],
        'rate': 0.4,
        'N': shape[0] * shape[1],
        'restart_fista': False,
        'err_criterion': False, # Criterion in ex. 2.3.b
        'stopping_criterion': lambda err_k, err_0: err_k / err_0, # Criterion in 2.3.b
        'f_star_criterion': False # Criterion in ex. 2.3.c and d
    }

    PATH = 'data/me.jpg'
    image = load_image(PATH, params['shape'])

    im_us, mask = apply_random_mask(image, params['rate'])
    indices = np.nonzero(mask.flatten(order='F'))[0]
    params['indices'] = indices

    # First check if FISTA works on just one lambda (= 0.01)
    ex_2_3_a = False
    if ex_2_3_a:
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
        ax[2].set_title(
            'L1 - PSNR = {:.2f}\n SSIM  = {:.2f} - Time: {:.2f}s'.format(psnr_l1, ssim_l1, t_l1))
        ax[3].imshow(reconstruction_tv, cmap="gray")
        ax[3].set_title(
            'TV - PSNR = {:.2f}\n SSIM  = {:.2f}  - Time: {:.2f}s'.format(psnr_tv, ssim_tv, t_tv))
        [axi.set_axis_off() for axi in ax.flatten()]
        plt.tight_layout()
        plt.show()

    # Create arrays to store psnrs
    psnr_l1_array = []
    psnr_tv_array = []

    # Set a ligarithmic spaced search space, from 1e-5 to 10, with 15 steps
    search_space = np.logspace(-5, 1, num=15)

    # Search for the best lambda
    search_lambda = False
    if search_lambda:

        print('\n----- Ex. 2.2.b ------\n')
        # Iteraye over the search space
        for lmbd in search_space:
            params['lambda'] = lmbd

            # Reconstruct with L-1 norm and get corresponding psnr
            reconstruction_l1 = reconstruct_l1(
                image, indices, FISTA, params)[0]
            psnr_l1 = psnr(image, reconstruction_l1)

            # Reconstruct with TV norm and get corresponding psnr
            reconstruction_tv = reconstruct_TV(
                image, indices, FISTA, params)[0]
            psnr_tv = psnr(image, reconstruction_tv)

            # Append results to the arrays
            psnr_l1_array.append(psnr_l1)
            psnr_tv_array.append(psnr_tv)

        # Ex 2.2.c
        # Plot psnr as a function of lambda
        plt.plot(search_space, psnr_l1_array, label='l1 reconstruction')
        plt.plot(search_space, psnr_tv_array,
                 color='r', label='TV reconstruction')
        plt.xscale('log')
        plt.xticks(rotation=60)
        plt.xlabel('$\lambda$')
        plt.ylabel('PSNR')
        plt.legend()
        plt.show()

        # Get the index of the best lambdas (where psnr is the highest)
        best_l1_lambda_index = np.argmax(psnr_l1_array)
        best_tv_lambda_index = np.argmax(psnr_tv_array)

        # Get the best lambdas
        best_l1_lambda = search_space[best_l1_lambda_index]
        best_tv_lambda = search_space[best_tv_lambda_index]

        print(f'Best l1 lambda = {best_l1_lambda}')
        print(f'Best tv lambda = {best_tv_lambda}')

    # Ex. 2.3.b
    ex_2_3_b = False
    if ex_2_3_b:
        print('\n----- Ex. 2.3.b ------\n')
        # Set corresponding parameters
        params['lambda'] = 0.01
        params['maxit'] = 5000
        params['restart_fista'] = True

        # Activate crietrion described in ex. 2.3.b
        params['err_criterion'] = True

        # Run fista with the params defined above
        reconstruction_l1 = reconstruct_l1(image, indices, FISTA, params)

        # Get the real number of iterations being done
        # before the criterion got satisfied
        total_iters = reconstruction_l1[1]['iters']

        # Get the corresponding f*
        f_star = reconstruction_l1[1]['conv'][total_iters]

        print(f'f* = {f_star}')

        # Set params for ex. 2.3.c-d
        params['f_star'] = f_star

        # Remove criterion described in ex. 2.3.b
        params['err_criterion'] = False

        # Set criterion described in ex. 2.3.c and define
        # corresponding function
        params['f_star_criterion'] = True
        params['stopping_criterion'] = lambda F_x, F_star: np.abs(
            F_x - F_star) / F_star

    # Ex 2.3.c
    ex_2_3_c = False
    if ex_2_3_c:
        print('\n----- Ex. 2.3.c ------\n')
        params['maxit'] = 2000

        # Run all methods using the new criterion and the found f*
        results = call_all_methods(image, indices, params, reconstruct_l1)

        # Set correct label for plot, and plot convergence results
        ylabel = r'$|f(\mathbf{x}^k) - f^\star| / f^\star$'
        plot_convergence(results, f_star, 'L-1 convergence', ylabel)

    # Ex 2.3.d
    ex_2_3_d = False
    if ex_2_3_d:
        print('\n----- Ex. 2.3.d ------\n')

        # Compute the ground truth as the L-1 norm of the
        # wavelet transform of the original image
        f_natural = params['lambda'] * \
            np.linalg.norm(r.W(image.reshape(-1)), 1)

        print(f'f_natural = {f_natural}')

        params['maxit'] = 1000
        params['f_star'] = f_natural

        # Rerun using the grounf truth as f*
        results = call_all_methods(image, indices, params, reconstruct_l1)

        # Set correct label for plot, and plot convergence results
        ylabel = r'$|f(\mathbf{x}^k) - f^\natural| / f^\natural$'
        plot_convergence(results, f_natural, 'L-1 convergence', ylabel)

    # Ex 2.4.a
    ex_2_4_a = True
    if ex_2_4_a:

        params['maxit'] = 500
        params['f_star_criterion'] = False
        params['restart_fista'] = True

        compare_unrolled(params, image, indices)

    # Ex 2.4.b
    ex_2_4_b = True
    if ex_2_4_b:

        params['maxit'] = 5

        compare_unrolled(params, image, indices)
