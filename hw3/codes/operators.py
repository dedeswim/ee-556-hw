from numpy import sum, exp, log, diag, multiply, maximum, absolute, sign, transpose, dot
import numpy as np
import pywt
import pywt.data


##########################################################################
# Operators for Exercise 1
##########################################################################
from random import randint


def l1_prox(y, weight):
    ############## YOUR CODES HERE ##############
    raise NotImplemented('Implement the method!')


def l2_prox(y, weight):
    ############## YOUR CODES HERE ##############
    raise NotImplemented('Implement the method!')


def gradfx(X, A, b):
    ############## YOUR CODES HERE ##############
    raise NotImplemented('Implement the method!')


def stocgradfx(X, minibatch_size, A, b):
    ############## YOUR CODES HERE ##############
    raise NotImplemented('Implement the method!')


def fx(X, A, b):
    num_samples = A.shape[0]
    return sum(log(sum(exp(A @ X), axis=1)), axis=0) \
           - sum([dot(A[i, :], X[:, b[i]]) for i in range(0, num_samples)])


def norm1(X):
    return np.linalg.norm(X.flatten('F'), 1)


def norm2sq(X):
    return (1.0 / 2) * np.linalg.norm(X, 'fro') ** 2


##########################################################################
# Operators for Exercise 2
##########################################################################

def TV_norm(X, opt=None):
    """
        Computes the TV-norm of image X
        opts = 'iso' for isotropic, else it is the anisotropic TV-norm
    """

    m, n = X.shape
    P1 = X[0:m - 1, :] - X[1:m, :]
    P2 = X[:, 0:n - 1] - X[:, 1:n]

    if opt == 'iso':
        D = np.zeros_like(X)
        D[0:m - 1, :] = P1 ** 2
        D[:, 0:n - 1] = D[:, 0:n - 1] + P2 ** 2
        tv_out = np.sum(np.sqrt(D))
    else:
        tv_out = np.sum(np.abs(P1)) + np.sum(np.abs(P2))

    return tv_out


# P_Omega and P_Omega_T
def p_omega(x, indices):  # P_Omega

    return np.expand_dims(x[np.unravel_index(indices, x.shape)], 1)


def p_omega_t(x, indices, m):  # P_Omega^T
    y = np.zeros((m, m))
    y[np.unravel_index(indices, y.shape)] = x.squeeze()
    return y


class RepresentationOperator(object):
    """
        Representation Operator contains the forward and adjoint
        operators for the Wavelet transform.
    """

    def __init__(self, m=256):
        self.m = m
        self.N = m ** 2

        self.W_operator = lambda x: pywt.wavedec2(x, 'db8', mode='periodization')  # From image coefficients to wavelet
        self.WT_operator = lambda x: pywt.waverec2(x, 'db8', mode='periodization')  # From wavelet coefficients to image
        _, self.coeffs = pywt.coeffs_to_array(self.W_operator(np.ones((m, m))))

    def W(self, x):
        """
            Computes the Wavelet transform from a vectorized image.
        """
        x = np.reshape(x, (self.m, self.m))
        wav_x, _ = pywt.coeffs_to_array(self.W_operator(x))

        return np.reshape(wav_x, (self.N, 1))

    def WT(self, wav_x):
        """
            Computes the adjoint Wavelet transform from a vectorized image.
        """
        wav_x = np.reshape(wav_x, (self.m, self.m))
        x = self.WT_operator(pywt.array_to_coeffs(wav_x, self.coeffs, output_format='wavedec2'))
        return np.reshape(x, (self.N, 1))
