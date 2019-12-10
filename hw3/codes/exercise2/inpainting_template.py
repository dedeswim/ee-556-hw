import sys
sys.path.append('../')

from operators import TV_norm, RepresentationOperator, p_omega, p_omega_t, l1_prox
from utils import print_end_message, print_start_message, print_progress, apply_random_mask, psnr, load_image
from plot_utils import plot_convergence
from skimage.measure import compare_ssim as ssim
from skimage.restoration import denoise_tv_chambolle
import matplotlib.pyplot as plt
import numpy as np
import time


def ISTA(fx, gx, gradf, proxg, params):
    method_name = 'ISTA'
    print_start_message(method_name)

    tic = time.time()

    lmbd = params['lambda']
    maxit = params['maxit']
    x = params['x0']
    alpha = 1 / params['Lips']

    run_details = {'X_final': None, 'conv': np.zeros(params['maxit'] + 1)}
    run_details['conv'][0] = fx(params['x0']) + gx(params['x0'])

    for k in range(maxit):

        x_next = proxg(x - alpha * gradf(x), alpha)

        # Record convergence
        run_details['conv'][k] = fx(x) + gx(x)

        if k % params['iter_print'] == 0:
            print_progress(k, params['maxit'],
                           run_details['conv'][k], fx(x), gx(x))

        stopping_criterion = params['stopping_criterion']
        tol = params['tol']
        if params['f_star_criterion']:
            f_star = params['f_star']
            f_k = run_details['conv'][k]
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
        x_next = proxg(y - alpha * gradf(y), alpha)
        t_next = (1 + np.sqrt(4 * (t ** 2) + 1)) / 2
        y_next = x_next + (t - 1) / t_next * (x_next - x)

        if params['restart_fista'] and gradient_scheme_restart_condition(x, x_next, y):
            print('Restarting...')
            t_next = 1
            y_next = x_next
            x_next = proxg(y_next - alpha * gradf(y_next), alpha)

        # Record convergence
        run_details['conv'][k] = fx(x) + gx(x)

        if k % params['iter_print'] == 0:
            print_progress(k, params['maxit'],
                           run_details['conv'][k], fx(x_next), gx(x_next))

        if params['err_criterion']:
            if k == 0:
                err_0 = np.linalg.norm(y_next - y)
            else:
                err_k = np.linalg.norm(y_next - y)
                if stopping_criterion(err_k, err_0) < tol:
                    break

        if params['f_star_criterion']:
            f_star = params['f_star']
            f_k = run_details['conv'][k]
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

    # Generate measurements
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

    # Generate measurements
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

# %%


if __name__ == "__main__":

    ##############################
    # Load image and sample mask #
    ##############################
    r = RepresentationOperator()
    shape = (256, 256)

    params = {
        'maxit': 200,
        'tol': 1e-15,
        'Lips': 1,
        'lambda': 0.01,
        'x0': np.zeros((shape[0] * shape[1], 1)),
        'restart_criterion':  gradient_scheme_restart_condition,
        'stopping_criterion': lambda err_k, err_0: err_k / err_0,
        'iter_print': 50,
        'shape': shape,
        'restart_param': 50,
        'verbose': True,
        'm': shape[0],
        'rate': 0.4,
        'N': shape[0] * shape[1],
        'restart_fista': False,
        'err_criterion': False,
        'f_star_criterion': False
    }

    PATH = 'data/me.jpg'
    image = load_image(PATH, params['shape'])

    im_us, mask = apply_random_mask(image, params['rate'])
    indices = np.nonzero(mask.flatten(order='F'))[0]
    params['indices'] = indices
    # Choose optimization parameters

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

    psnr_l1_array = []
    psnr_tv_array = []
    search_space = np.logspace(-5, 1, num=15)

    search_lambda = False
    if search_lambda:

        print('\n----- Ex. 2.2.b ------\n')
        for lmbd in search_space:
            params['lambda'] = lmbd

            reconstruction_l1 = reconstruct_l1(
                image, indices, FISTA, params)[0]
            psnr_l1 = psnr(image, reconstruction_l1)

            reconstruction_tv = reconstruct_TV(
                image, indices, FISTA, params)[0]
            psnr_tv = psnr(image, reconstruction_tv)

            psnr_l1_array.append(psnr_l1)
            psnr_tv_array.append(psnr_tv)

        # Ex 2.2.c
        plt.plot(search_space, psnr_l1_array, label='l1 reconstruction')
        plt.plot(search_space, psnr_tv_array,
                 color='r', label='TV reconstruction')
        plt.xscale('log')
        plt.xticks(rotation=60)
        plt.xlabel('$\lambda$')
        plt.ylabel('PSNR')
        plt.legend()
        plt.show()

        best_l1_lambda_index = np.argmax(psnr_l1_array)
        best_l1_lambda = search_space[best_l1_lambda_index]

        best_tv_lambda_index = np.argmax(psnr_tv_array)
        best_tv_lambda = search_space[best_tv_lambda_index]

        print(f'Best l1 lambda = {best_l1_lambda}')
        print(f'Best tv lambda = {best_tv_lambda}')

    # Ex. 2.3.b
    print('\n----- Ex. 2.3.b ------\n')
    params['lambda'] = 0.01
    params['maxit'] = 5000
    params['restart_fista'] = True
    params['err_criterion'] = True

    reconstruction_l1 = reconstruct_l1(image, indices, FISTA, params)
    total_iters = reconstruction_l1[1]['iters']
    f_star = reconstruction_l1[1]['conv'][total_iters]

    print(f'f* = {f_star}')

    # Set params for ex. 2.3.c-d
    params['f_star'] = f_star
    params['err_criterion'] = False
    params['f_star_criterion'] = True
    params['stopping_criterion'] = lambda F_x, F_star: np.abs(F_x - F_star) / F_star
    
    # Ex 2.3.c
    ex_2_3_c = False
    if ex_2_3_c:
        print('\n----- Ex. 2.3.c ------\n')        
        params['maxit'] = 2000
        ylabel = r'$|f(\mathbf{x}^k) - f^\star| / f^\star$'
        results = call_all_methods(image, indices, params, reconstruct_l1)
        plot_convergence(results, f_star, 'L-1 convergence', ylabel)

    # Ex 2.3.d
    ex_2_3_d = True
    if ex_2_3_d:
        print('\n----- Ex. 2.3.d ------\n')
        
        # Define the overall operator
        def forward_operator(x): return p_omega(r.WT(x), indices)  # P_Omega.W^T

        # Generate measurements
        b = p_omega(image.reshape(-1), indices)

        f_natural = params['lambda'] * np.linalg.norm(r.W(image), 1)
        print(f'f_natural = {f_natural}')

        params['maxit'] = 1000
        params['f_star'] = f_natural
        ylabel = r'$|f(\mathbf{x}^k) - f^\natural| / f^\natural$'
        results = call_all_methods(image, indices, params, reconstruct_l1)
        plot_convergence(results, f_natural, 'L-1 convergence', ylabel)
