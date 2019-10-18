import numpy as np

#  compute f
def fsmoothedhinge(A,b,x):
    fx = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        ci = b[i]*np.dot(A[i],x)
        if ci <= 0:
            fx[i] = 0.5 - ci
        elif ci <= 1:
            fx[i] = 0.5*(1-ci)**2
        else:
            fx[i] = 0
    fx = fx.mean()
    return fx

# compute gradient
def gradfsmoothedhinge(A,b,x):
    n = A.shape[0]
    gradfx = np.zeros(x.shape)
    for i in range(n):
        ci = np.dot(b[i], np.dot(A[i],x))
        if ci <= 0:
            gradfx += -b[i] * A[i]
        elif ci <= 1:
            gradfx += (ci-1)*b[i]*A[i]
    gradfx = gradfx/n

    return gradfx

def stogradfsmoothedhinge(A,b,x, i):
    ci = np.dot(b[i], np.dot(A[i],x))
    if ci <= 0:
        gradfx = -b[i] * A[i]
    elif ci <= 1:
        gradfx = (ci-1)*b[i]*A[i]
    else:
        gradfx = np.zeros(x.shape)
    return gradfx

def Oracles(b, A, lbd):
    """
    FIRSTORDERORACLE
    Takes inputs b, A, sigma and returns two anonymous functions, one for
    the objective evaluation and the other for the gradient.
    fx computes the objective (l-2 regularized) of input x
    gradf computes the gradient (l-2 regularized) of input x
    """
    n, p = A.shape
    fx  = lambda x : 0.5*lbd*np.linalg.norm(x, 2)**2 + fsmoothedhinge(A,b,x)
    gradf  = lambda x: lbd*x + gradfsmoothedhinge(A,b,x)
    gradfsto = lambda x, i: lbd * x + stogradfsmoothedhinge(A, b, x, i)
    return fx, gradf, gradfsto

# def stofirstOrderOracle(b, A, lbd):
#     """
#     FIRSTORDERORACLE
#     Takes inputs b, A, sigma and returns two anonymous functions, one for
#     the objective evaluation and the other for the gradient.
#     fx computes the objective (l-2 regularized) of input x
#     gradf computes the gradient (l-2 regularized) of input x
#     """
#     m = b.shape[0]
#     fx  = lambda x : 0.5*lbd*np.linalg.norm(x, 2)**2 + fsquaredhinge(A,b,x)
#     gradf  = lambda x, i: lbd*m*x + stogradfsquaredhinge(A,b,x, i)
#     return fx, gradf

# def secondOrderOracle( b, A, lbd):
#     """
#     SECONDORDERORACLE
#     Takes inputs b, A, sigma and returns three anonymous functions, one for
#     the objective evaluation, one for the gradient, and the last for the
#     Hessian.
#     :param b:
#     :param A:
#     :param lbd:
#     :return: fx, gradf, hessfx
#     """
#
#     fx = lambda x: 0.5 * lbd * np.linalg.norm(x, 2) ** 2 + fsquaredhinge(A, b, x)
#     gradf = lambda x: lbd * x + gradfsquaredhinge(A, b, x)
#     p  = A.shape[1]
#     hessfx  = lambda x: lbd*np.identity(p) + hessfsquaredhinge(A,b,x)
#
#     return fx, gradf, hessfx

def compute_error(A_test,b_test,x):
    n, err = A_test.shape[0], 0
    for i in range(n):
        if np.dot(b_test[i],np.dot(A_test[i],x)) <= 0:
           err += 1
    err = err/float(n)
    return err
