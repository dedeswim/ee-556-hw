import time
import numpy as np
from random import randint

from utils import print_end_message, print_start_message, print_progress

def ista(fx, gx, gradf, proxg, params):
    method_name = 'ISTA'
    print_start_message(method_name)

    tic = time.time()

    run_details = {'X_final': None, 'conv': np.zeros(params['maxit'] + 1)}
    run_details['conv'][0] = fx(params['x0']) + lmbd * gx(params['x0'])

    ############## YOUR CODES HERE - parameter setup##############

    for k in range(############## YOUR CODES HERE ############## ):

        ############## YOUR CODES HERE ##############

        if k % params['iter_print'] == 0:
            print_progress(k, params['maxit'], run_details['conv'][k], fx(), gx())


    run_details['X_final'] = ############## YOUR CODES HERE ##############

    print_end_message(method_name, time.time() - tic)
    return run_details




def fista(fx, gx, gradf, proxg, params):
    if params['restart_fista']:
        method_name = 'FISTA-RESTART'
    else:
        method_name = 'FISTA'
    print_start_message(method_name)

    run_details = {'X_final': None, 'conv': np.zeros(params['maxit'] + 1)}
    run_details['conv'][0] = fx(params['x0']) + lmbd * gx(params['x0'])

    ############## YOUR CODES HERE - parameter setup##############

    for k in range(############## YOUR CODES HERE - parameter setup##############):
            ############## YOUR CODES HERE##############

        # record convergence
        run_details['conv'][k] = fx() + lmbd * gx()

        if k % params['iter_print'] == 0:
            print_progress(k, params['maxit'], run_details['conv'][k], fx(), gx())

    run_details['X_final'] = ############## YOUR CODES HERE##############

    print_end_message(method_name, time.time() - tic)
    return run_details




def gradient_scheme_restart_condition(X_k, X_k_next, Y_k):
    ############## YOUR CODES HERE ##############
    raise NotImplemented('Implement the method!')





def prox_sg(fx, gx, stocgradfx, proxg, params):
    method_name = 'PROX-SG'
    print_start_message(method_name)

    tic = time.time()


    run_details = {'X_final': None, 'conv': np.zeros(params['maxit'] + 1)}
    run_details['conv'][0] = fx(params['x0']) + lmbd * gx(params['x0'])

    ############## YOUR CODES HERE - parameter setup##############
    for k in range(############## YOUR CODES HERE ##############):
            ############## YOUR CODES HERE ##############

        run_details['conv'][k] = fx() + lmbd * gx()

        if k % params['iter_print'] == 0:
            print_progress(k, params['maxit'], run_details['conv'][k], fx(), gx())

    run_details['X_final'] = ############## YOUR CODES HERE ##############

    print_end_message(method_name, time.time() - tic)
    return run_details
