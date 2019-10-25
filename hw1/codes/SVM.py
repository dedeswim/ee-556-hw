import numpy as np
from numpy import linalg as LA
from SVM.commons import Oracles, compute_error
from SVM.algorithms import  GD, GDstr, AGD, AGDstr, LSGD, LSAGD, AGDR, LSAGDR, AdaGrad, ADAM, SGD, SAG, SVR
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

def main():
    training, testing = np.load('dataset/training.npz'), np.load('dataset/testing.npz')
    A, b = training['A'], training['b']
    A_test, b_test = testing['A'], testing['b']

    print(68 * '*')
    print('Linear Support Vector Machine:')
    print('Smoothed Hinge Loss + Ridge regularizer')
    print('dataset size : {} x {}'.format(A.shape[0], A.shape[1]))
    print(68 * '*')

    # Choose the solvers you want to call
    GD_option = 0
    GDstr_option = 0
    AGD_option = 0
    AGDstr_option = 0
    LSGD_option = 0
    LSAGD_option = 0
    AGDR_option = 1
    LSAGDR_option = 0
    AdaGrad_option = 0
    ADAM_option = 0
    SGD_option = 0
    SAG_option = 0
    SVR_option = 0


    # Set parameters and solve numerically with GD, AGD, AGDR, LSGD, LSAGD, LSAGDR.
    print('Numerical solution process is started:')
    n, p = A.shape
    sigma = 1e-4
    fs_star = 0.039897199
    parameter = {}
    parameter['Lips'] = LA.norm(np.transpose(A), 2)*LA.norm(A, 2)/n+ sigma
    parameter['strcnvx'] = sigma
    parameter['x0'] = np.zeros(p)
    parameter['Lmax'] = 0
    for i in range(n):
        parameter['Lmax']= np.maximum(parameter['Lmax'],LA.norm(A[i], 2)*LA.norm(A[i], 2))
    parameter['Lmax'] += sigma

    fx, gradf, gradfsto = Oracles(b,A,sigma)
    x, info, error = {}, {}, {}

    # first-order methods
    parameter['maxit'] = 4000
    if GD_option:
        x['GD'], info['GD'] = GD(fx, gradf, parameter)
        error['GD'] = compute_error(A_test, b_test, x['GD'])
        print('Error w.r.t 0-1 loss: {}'.format(error['GD']))

    parameter['maxit'] = 4000
    if GDstr_option:
        x['GDstr'], info['GDstr'] = GDstr(fx, gradf, parameter)
        error['GDstr'] = compute_error(A_test, b_test, x['GDstr'])
        print('Error w.r.t 0-1 loss: {}'.format(error['GDstr']))

    parameter['maxit'] = 2300
    if AGD_option:
        x['AGD'], info['AGD'] = AGD(fx, gradf, parameter)
        error['AGD'] = compute_error(A_test, b_test, x['AGD'])
        print('Error w.r.t 0-1 loss: {}'.format(error['AGD']))

    parameter['maxit'] = 1700
    if AGDstr_option:
        x['AGDstr'], info['AGDstr'] = AGDstr(fx, gradf, parameter)
        error['AGDstr'] = compute_error(A_test, b_test, x['AGDstr'])
        print('Error w.r.t 0-1 loss: {}'.format(error['AGDstr']))

    parameter['maxit'] = 300
    if AGDR_option:
        x['AGDR'], info['AGDR'] = AGDR(fx, gradf, parameter)
        error['AGDR'] = compute_error(A_test, b_test, x['AGDR'])
        print('Error w.r.t 0-1 loss: {}'.format(error['AGDR']))

    parameter['maxit'] = 500
    if LSGD_option:
        x['LSGD'], info['LSGD'] = LSGD(fx, gradf, parameter)
        error['LSGD'] = compute_error(A_test, b_test, x['LSGD'])
        print('Error w.r.t 0-1 loss: {}'.format(error['LSGD']))

    if LSAGD_option:
        x['LSAGD'], info['LSAGD'] = LSAGD(fx, gradf, parameter)
        error['LSAGD'] = compute_error(A_test, b_test, x['LSAGD'])
        print('Error w.r.t 0-1 loss: {}'.format(error['LSAGD']))

    parameter['maxit'] = 100
    if LSAGDR_option:
        x['LSAGDR'], info['LSAGDR'] = LSAGDR(fx, gradf, parameter)
        error['LSAGDR'] = compute_error(A_test, b_test, x['LSAGDR'])
        print('Error w.r.t 0-1 loss: {}'.format(error['LSAGDR']))
    
    parameter['maxit'] = 4000
    if AdaGrad_option:
        x['AdaGrad'], info['AdaGrad'] = AdaGrad(fx, gradf, parameter)
        error['AdaGrad'] = compute_error(A_test, b_test, x['AdaGrad'])
        print('Error w.r.t 0-1 loss: {}'.format(error['AdaGrad']))
    
    parameter['maxit'] = 4000
    if ADAM_option:
        x['ADAM'], info['ADAM'] = ADAM(fx, gradf, parameter)
        error['ADAM'] = compute_error(A_test, b_test, x['ADAM'])
        print('Error w.r.t 0-1 loss: {}'.format(error['ADAM']))

    # stochastic methods
    parameter['no0functions'] = n
    parameter['maxit'] = 5*n
    if SGD_option:
        x['SGD'], info['SGD'] = SGD(fx, gradfsto, parameter)
        error['SGD'] = compute_error(A_test, b_test, x['SGD'])
        print('Error w.r.t 0-1 loss: {}'.format(error['SGD']))

    if SAG_option:
        x['SAG'], info['SAG'] = SAG(fx, gradfsto, parameter)
        error['SAG'] = compute_error(A_test, b_test, x['SAG'])
        print('Error w.r.t 0-1 loss: {}'.format(error['SAG']))

    parameter['maxit'] = int(1.5*n)
    if SVR_option:
        x['SVR'], info['SVR'] = SVR(fx, gradf, gradfsto, parameter)
        error['SVR'] = compute_error(A_test, b_test, x['SVR'])
        print('Error w.r.t 0-1 loss: {}'.format(error['SVR']))


    print('Numerical solution process is completed.')
    colors = [(0, 0, 1),(0, 0.5, 0),(1, 0, 0),(0, 0.75, 0.75),(0.75, 0, 0.75),(0.75, 0.75, 0),(0, 0, 0),(0.5,1,0.5),(0.5,0,0.5),(0.75,0.25,0.25)]

    # plot figures
    ax1 = plt.subplot(1, 2, 1)

    for i, key in enumerate(x.keys()):
        if key not in ['SGD', 'SAG', 'SVR']:
            ax1.plot(np.array(range(1, info[key]['iter']+1)), info[key]['fx'] - fs_star, color=colors[i], lw=2, label=key)
    ax1.legend()
    ax1.set_ylim(1e-9, 1e0)
    ax1.set_xlabel('#iterations')
    ax1.set_ylabel(r'$f(\mathbf{x}^k) - f^\star$')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid()

    # plt.subplot(2, 2, 2)
    #
    # for key in x.keys():
    #     if key not in ['SGD', 'SAG', 'SVR']:
    #         plt.plot(np.cumsum(info[key]['itertime']), info[key]['fx'] - fs_star, lw=2, label=key)
    #
    # plt.xlabel('time (s)')
    # plt.ylabel('f(x) - f*')
    # plt.legend()
    # plt.xscale('log', nonposy='clip')
    # plt.yscale('log', nonposy='clip')

    ax2 = plt.subplot(1, 2, 2)

    for key in x.keys():
        if key in ['GD', 'SGD', 'SAG', 'SVR']:
            if key =='GD':
                ax2.plot(np.array(range(info[key]['iter'])), info[key]['fx'] - fs_star, lw=2, label=key, marker='o')
            else:
                ax2.plot(np.array(range(info[key]['iter']))/float(n),info[key]['fx'] - fs_star, lw=2, label=key)
    ax2.set_xlim(0, 5)
    ax2.set_ylim(1e-4, 1e0)
    ax2.legend()
    ax2.set_xlabel('#epochs')
    ax2.set_ylabel(r'$f(\mathbf{x}^k) - f^\star$')
    # ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid()

    # plt.subplot(2, 2, 4)
    #
    # for key in x.keys():
    #     if key in ['GD', 'SGD', 'SAG', 'SVR']:
    #         plt.plot(np.cumsum(info[key]['itertime']), info[key]['fx'] - fs_star, lw=2, label=key)
    #
    # plt.xlabel('time (s)')
    # plt.ylabel('f(x) - f*')
    # plt.legend()
    # plt.xscale('log', nonposy='clip')
    # plt.yscale('log', nonposy='clip')

    plt.tight_layout()
    plt.savefig('fig_ex1.pdf')
    # plt.show()

if __name__ == "__main__":
    main()


