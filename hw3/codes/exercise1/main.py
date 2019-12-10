import sys
sys.path.append('../')

from utils import *
from algorithms import fista, ista, prox_sg
from plot_utils import plot_convergence, plot_digit_features
from operators import l1_prox, l2_prox, norm1, norm2sq
import operators
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import time


RAND_SEED = 666013


def call_all_methods(fx, gx, gradfx, stocgradfx, prox_fc, params):
    all_results = dict()

    params['maxit'] = params['maxit_determ']
    all_results['ISTA'] = ista(fx, gx, gradfx, prox_fc, params)

    params['restart_fista'] = False
    all_results['FISTA'] = fista(fx, gx, gradfx, prox_fc, params)

    params['restart_fista'] = True
    all_results['FISTA-RESTART'] = fista(fx, gx, gradfx, prox_fc, params)

    params['maxit'] = params['maxit_stoch']
    all_results['PROX-SG'] = prox_sg(fx, gx, stocgradfx, prox_fc, params)

    return all_results


def main():
    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)

    A_train, b_train, b_train_binarized, A_test, b_test, num_classes, num_features = load_mnist()
    lmbd_l1 = 1.0
    lmbd_l2 = 10.0

    def fx(Y): return operators.fx(Y, A_train, b_train)
    def gradfx(Y): return operators.gradfx(Y, A_train, b_train_binarized)
    def stocgradfx(Y, minimabtch_size): return operators.stocgradfx(
        Y, minimabtch_size, A_train, b_train_binarized)

    Lips = (np.linalg.norm(A_train) ** 2) / 2

    f_star_l1, X_opt_l1, f_star_l2, X_opt_l2 = read_f_star(
        fx, lmbd_l1, lmbd_l2)

    params = {
        'maxit_determ': 2000,
        'maxit_stoch': 50000,
        'maxit': None,
        'Lips':  Lips,
        'lambda': None,
        'x0': np.random.rand(num_features, num_classes),
        'restart_fista': False,
        'iter_print': 100,
        'verbose': True,
        'minib_size': 100,
        'stoch_rate_regime': lambda k: 1/(Lips + k**(0.55)),
        'n': A_train.shape[0]
    }
    epoch_to_iteration_exchange_rate = int(params['n']/params['minib_size'])
    print(Lips)

    params['lambda'] = lmbd_l1
    all_results_l1 = call_all_methods(
        fx, operators.norm1, gradfx, stocgradfx, operators.l1_prox, params)
    plot_convergence(all_results_l1, f_star_l1, epoch_to_iteration_exchange_rate,
                     'L1 - regularized LogisticRegression')
    plot_digit_features(all_results_l1['FISTA-RESTART']['X_final'],
                        'Visualization of solution for L1 - regularized LogisticRegression')
    print('FISTA-RESTART-l1 accuracy = {:f}%.\n'.format(compute_accuracy(
        all_results_l1['FISTA-RESTART']['X_final'], A_test, b_test) * 100))

    params['lambda'] = lmbd_l2
    all_results_l2 = call_all_methods(
        fx, operators.norm2sq, gradfx, stocgradfx, operators.l2_prox, params)
    plot_convergence(all_results_l2, f_star_l2, epoch_to_iteration_exchange_rate,
                     'L2 - regularized LogisticRegression')
    plot_digit_features(all_results_l2['FISTA-RESTART']['X_final'],
                        'Visualization of solution for L2 - regularized LogisticRegression')

    print('FISTA-RESTART-l2 accuracy = {:f}%.\n'.format(
        compute_accuracy(all_results_l2['FISTA-RESTART']['X_final'], A_test, b_test) * 100))

    print('============= THAT\'S ALL FOLKS! =============')


if __name__ == "__main__":
    main()
