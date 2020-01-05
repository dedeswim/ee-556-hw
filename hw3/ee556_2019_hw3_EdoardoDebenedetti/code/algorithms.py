import time
import numpy as np
from random import randint

from utils import print_end_message, print_start_message, print_progress


def ista(fx, gx, gradf, proxg, params):
    method_name = 'ISTA'
    print_start_message(method_name)

    tic = time.time()

    # Take parameters
    lmbd = params['lambda']
    maxit = params['maxit']
    x = params['x0']
    alpha = 1 / params['Lips']

    run_details = {'X_final': None, 'conv': np.zeros(params['maxit'] + 1)}
    run_details['conv'][0] = fx(params['x0']) + lmbd * gx(params['x0'])

    for k in range(maxit):

        # Compute the prox
        x_next = proxg(x - alpha * gradf(x), alpha * lmbd)

        # Record convergence
        run_details['conv'][k] = fx(x) + lmbd * gx(x)

        if k % params['iter_print'] == 0:
            print_progress(k, params['maxit'],
                           run_details['conv'][k], fx(x), gx(x))
        
        x = x_next

    run_details['conv'][k + 1] = fx(x) + lmbd * gx(x)

    run_details['X_final'] = x

    print_end_message(method_name, time.time() - tic)
    return run_details


def fista(fx, gx, gradf, proxg, params):
    if params['restart_fista']:
        method_name = 'FISTA-RESTART'
    else:
        method_name = 'FISTA'
    print_start_message(method_name)

    tic = time.time()

    # Get parameters
    lmbd = params['lambda']
    maxit = params['maxit']
    alpha = 1 / params['Lips']

    # Initialize variables
    x = params['x0']
    y = x
    t = 1

    run_details = {'X_final': None, 'conv': np.zeros(params['maxit'] + 1)}
    run_details['conv'][0] = fx(params['x0']) + lmbd * gx(params['x0'])

    for k in range(maxit):
        # Compute prox
        x_next = proxg(y - alpha * gradf(y), lmbd * alpha)

        # Compute the momentum
        t_next = (1 + np.sqrt(4 * (t ** 2) + 1)) / 2
        y_next = x_next + (t - 1) / t_next * (x_next - x)

        # If FISTA Restart AND condition is satisfied, restart
        if params['restart_fista'] and gradient_scheme_restart_condition(x, x_next, y):
            print('Restarting...')
            # Reset momentum
            t_next = 1
            y = x_next

            # Re-compute prox
            x_next = proxg(y - alpha * gradf(y), alpha * lmbd)

        # Record convergence
        run_details['conv'][k] = fx(x) + lmbd * gx(x)

        if k % params['iter_print'] == 0:
            print_progress(k, params['maxit'],
                           run_details['conv'][k], fx(x_next), gx(x_next))

        x = x_next
        y = y_next
        t = t_next

    run_details['conv'][k + 1] = fx(x) + lmbd * gx(x)

    run_details['X_final'] = x

    print_end_message(method_name, time.time() - tic)
    return run_details


def gradient_scheme_restart_condition(X_k, X_k_next, Y_k):
    return np.trace((Y_k - X_k_next).T @ (X_k_next - X_k)) > 0


def prox_sg(fx, gx, stocgradfx, proxg, params):
    method_name = 'PROX-SG'
    print_start_message(method_name)

    tic = time.time()

    # Save parameters
    lmbd = params['lambda']
    maxit = params['maxit']
    alpha = 1 / params['Lips']
    minibatch_size = params['minib_size']
    lr_regime = params['stoch_rate_regime']
    x = params['x0']

    # Initialize variables for ergodic convergence
    x_sum = 0
    alpha_sum = 0

    run_details = {'X_final': None, 'conv': np.zeros(params['maxit'] + 1)}
    run_details['conv'][0] = fx(params['x0']) + lmbd * gx(params['x0'])

    for k in range(maxit):
        # Compute gradients
        x_next = proxg(x - alpha * stocgradfx(x, minibatch_size), lmbd * alpha)

        # Record ergodic convergence
        x_sum += x * alpha
        alpha_sum += alpha
        x_hat = x_sum / alpha_sum

        run_details['conv'][k] = fx(x_hat) + lmbd * gx(x_hat)

        if k % params['iter_print'] == 0:
            print_progress(k, params['maxit'],
                           run_details['conv'][k], fx(x_hat), gx(x_hat))

        # Update alpha
        alpha = lr_regime(k)

        x = x_next

    run_details['conv'][k+1] = fx(x_hat) + lmbd * gx(x_hat)
    run_details['X_final'] = x

    print_end_message(method_name, time.time() - tic)
    
    return run_details
