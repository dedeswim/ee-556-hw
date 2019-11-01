import time
import numpy as np
import scipy.sparse.linalg as spla
from numpy.random import randint
from scipy.sparse.linalg.dsolve import linsolve
from itertools import count


# Gradient Descent
def GD(fx, gradf, parameter):
    """
    Function:  [x, info] = GD(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
               strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    print(68*'*')
    print('Gradient Descent')

    # Get parameters
    maxit = parameter['maxit']
    x0 = parameter['x0']
    Lips = parameter['Lips']

    # Initialize x and alpha.
    alpha = 1 / Lips
    x = x0

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        x_next = x - alpha * gradf(x)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(
                iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
    info['iter'] = maxit
    return x, info


# Gradient Descent with strong convexity
def GDstr(fx, gradf, parameter):
    """
    Function:  GDstr(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
               strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return: x, info
    """

    print(68*'*')
    print('Gradient Descent  with strong convexity')

    # Get parameters
    maxit = parameter['maxit']
    x0 = parameter['x0']
    Lips = parameter['Lips']
    strcnvx = parameter['strcnvx']

    # Initialize x and alpha.
    x = x0
    alpha = 2 / (Lips + strcnvx)

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start timer
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        x_next = x - alpha * gradf(x)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(
                iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    return x, info


# Accelerated Gradient
def AGD(fx, gradf, parameter):
    """
    *******************  EE556 - Mathematics of Data  ************************
    Function:   AGD(fx, gradf, parameter)
    Purpose:    Implementation of the accelerated gradient descent algorithm.
    Parameter:  x0         - Initial estimate.
                maxit      - Maximum number of iterations.
                Lips       - Lipschitz constant for gradient.
    *************************** LIONS@EPFL ***********************************
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    print(68 * '*')
    print('Accelerated Gradient')

    # Get parameters
    maxit = parameter['maxit']
    x0 = parameter['x0']
    Lips = parameter['Lips']

    # Initialize x, alpha, y and t.
    t = 1
    alpha = 1 / Lips
    x = x0
    y = x0

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):

        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        t_next = (1 + np.math.sqrt(4 * (t ** 2))) / 2
        x_next = y - alpha * gradf(y)
        y = x_next + (t - 1) / t_next * (x_next - x)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(
                iter, info['fx'][iter]))

        # Prepare next iteration
        x = x_next
        t = t_next

    return x, info


# Accelerated Gradient with strong convexity
def AGDstr(fx, gradf, parameter):
    """
    *******************  EE556 - Mathematics of Data  ************************
    Function:   AGDstr(fx, gradf, parameter)
    Purpose:    Implementation of the accelerated gradient descent algorithm, assuming strong convexity.
    Parameter:  x0         - Initial estimate.
                maxit      - Maximum number of iterations.
                Lips       - Lipschitz constant for gradient.
                strcnvx	   - strong convexity parameter of f(x).
    *************************** LIONS@EPFL ***********************************
    :param fx:
    :param gradf:
    :param parameter:
    :return: x, info
    """
    print(68 * '*')
    print('Accelerated Gradient with strong convexity')

    # Get parameters
    maxit = parameter['maxit']
    x0 = parameter['x0']
    Lips = parameter['Lips']
    strcnvx = parameter['strcnvx']

    # Initialize x, y, alpha and gamma.
    alpha = 2 / (Lips + strcnvx)
    gamma = ((np.math.sqrt(Lips) - np.math.sqrt(strcnvx)) /
             (np.math.sqrt(Lips) + np.math.sqrt(strcnvx)))
    x = x0
    y = x

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):

        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        x_next = y - alpha * gradf(y)
        y = x_next + gamma * (x_next - x)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(
                iter, info['fx'][iter]))

        # Prepare next iteration
        x = x_next

    return x, info


# Line Search Gradient Descent
def LSGD(fx, gradf, parameter):
    """
    Function:   [x, info] = LSGD(fx, gradf, parameter)
    Purpose:    Implementation of the gradient descent with line-search.
    Parameter:  x0         - Initial estimate.
                maxit      - Maximum number of iterations.
                Lips       - Lipschitz constant for gradient.
    :param fx:
    :param gradf:
    :param parameter:
    :return: x, info
    """
    print(68 * '*')
    print('Gradient Descent with line search')

    # Get parameters
    maxit = parameter['maxit']
    x0 = parameter['x0']
    Lips = parameter['Lips']

    # Initialize x and L0.
    x = x0
    Lk_0 = Lips

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        Lk_0 = 1 / 2 * Lk_0
        d = gradf(x)

        # Line Search procedure
        i = line_search(Lk_0, fx, x, d)

        Lk_0 = (2 ** i) * Lk_0
        x_next = x - 1 / Lk_0 * d

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(
                iter, info['fx'][iter]))

        # Prepare next iteration
        x = x_next

    return x, info

# LSAGD


def LSAGD(fx, gradf, parameter):
    """
    Function:   [x, info] = LSAGD (fx, gradf, parameter)
    Purpose:    Implementation of AGD with line search.
    Parameter:  x0         - Initial estimate.
                maxit      - Maximum number of iterations.
                Lips       - Lipschitz constant for gradient.
    :param fx:
    :param gradf:
    :param parameter:
    :return:
        """
    print(68 * '*')
    print('Accelerated Gradient with line search')

    # Get parameters
    maxit = parameter['maxit']
    x0 = parameter['x0']
    Lips = parameter['Lips']

    # Initialize x, y and t and L0.
    x = x0
    y = x
    t = 1
    Lk_0 = Lips

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        Lk_old = Lk_0
        Lk_0 = 1 / 2 * Lk_0
        d = gradf(y)

        # Line Search procedure
        i = line_search(Lk_0, fx, y, d)

        L_k = (2 ** i) * Lk_0
        t_next = 0.5 * (1 + np.math.sqrt(1 + 4 *
                                         (L_k / Lk_old) * (t ** 2))) / 2
        x_next = y - 1 / L_k * d
        y = x_next + (t - 1) / t_next * (x_next - x)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(
                iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        t = t_next
        Lk_0 = L_k

    return x, info


# AGDR
def AGDR(fx, gradf, parameter):
    """
    Function:  [x, info] = AGDR (fx, gradf, parameter)
    Purpose:   Implementation of the AGD with adaptive restart.
    Parameter:  x0         - Initial estimate.
                maxit      - Maximum number of iterations.
                Lips       - Lipschitz constant for gradient.
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    print(68 * '*')
    print('Accelerated Gradient with restart')

    # Get parameters
    maxit = parameter['maxit']
    x0 = parameter['x0']
    Lips = parameter['Lips']

    # Initialize x, y, t and find the initial function value (fval).
    t = 1
    alpha = 1 / Lips
    x = x0
    y = x0
    fval = fx(x)

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        t_next = (1 + np.math.sqrt(4 * (t ** 2))) / 2
        x_next = y - alpha * gradf(y)
        y = x_next + (t - 1) / t_next * (x_next - x)
        
        # Evaluate the next f
        fval_next = fx(x_next)

        # Check restart conditions
        if fval_next > fval:
            y = x
            t = 1

            # Re-compute parameters after reset
            x_next = y - alpha * gradf(y)
            fval_next = fx(x_next)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(
                iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        t = t_next
        fval = fval_next

    return x, info


# LSAGDR
def LSAGDR(fx, gradf, parameter):
    """
    Function:   [x, info] = LSAGDR (fx, gradf, parameter)
    Purpose:    Implementation of AGD with line search and adaptive restart.
    Parameter:  x0         - Initial estimate.
                maxit      - Maximum number of iterations.
                Lips       - Lipschitz constant for gradient.
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    print(68 * '*')
    print('Accelerated Gradient with line search + restart')

    # Get parameters
    maxit = parameter['maxit']
    x0 = parameter['x0']
    Lips = parameter['Lips']

    # Initialize x, y and t and L0.
    x = x0
    y = x
    t = 1
    Lk_0 = Lips
    fval = fx(x)

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        Lk_old = Lk_0
        Lk_0 = 1 / 2 * Lk_0
        d = gradf(y)

        # Line Search procedure
        i = line_search(Lk_0, fx, y, d)

        # Get new parameters
        L_k = (2 ** i) * Lk_0
        t_next = 0.5 * (1 + np.math.sqrt(1 + 4 *
                                         (L_k / Lk_old) * (t ** 2))) / 2
        x_next = y - 1 / L_k * d
        y = x_next + (t - 1) / t_next * (x_next - x)
        
        # Evaluate the next f
        fval_next = fx(x_next)

        # Check restart conditions
        if fval_next > fval:
            y = x
            t = 1
            d = gradf(y)
            # Re-perform line search
            i = line_search(Lk_0, fx, y, d)

            # Re-compute parameters after reset
            L_k = (2 ** i) * Lk_0
            t_next = 0.5 * (1 + np.math.sqrt(1 + 4 *
                                             (L_k / Lk_old) * (t ** 2))) / 2
            x_next = y - 1 / L_k * d
            y = x_next + (t - 1) / t_next * (x_next - x)
            fval_next = fx(x_next)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(
                iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        t = t_next
        Lk_0 = L_k
        fval = fval_next

    return x, info


def AdaGrad(fx, gradf, parameter):
    """
    Function:   [x, info] = AdaGrad (fx, gradf, parameter)
    Purpose:    Implementation of the adaptive gradient method with scalar step-size.
    Parameter:  x0         - Initial estimate.
                maxit      - Maximum number of iterations.
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    print(68 * '*')
    print('Adaptive Gradient method')

    # Get parameters
    maxit = parameter['maxit']
    x0 = parameter['x0']

    # Initialize x, alpha, delta (and any other)
    x = x0
    alpha = 1
    delta = 1e-5
    Q = 0

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.clock()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        grad = gradf(x)
        
        # Compute adaptive parameters
        Q_next = Q + np.linalg.norm(grad) ** 2
        H_k = np.math.sqrt(Q_next + delta)

        # Compute the next x
        x_next = x - alpha * np.dot((1 / H_k) * np.eye(grad.shape[0]), grad)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(
                iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        Q = Q_next

    return x, info

# Newton


def ADAM(fx, gradf, parameter):
    """
    Function:   [x, info] = ADAM (fx, gradf, parameter)
    Purpose:    Implementation of ADAM.
    Parameter:  x0         - Initial estimate.
                maxit      - Maximum number of iterations.
    :param fx:
    :param gradf:
    :param hessf:
    :param parameter:
    :return:
    """

    print(68 * '*')
    print('ADAM')

    # Get parameters
    maxit = parameter['maxit']
    x0 = parameter['x0']

    # Initialize x, beta1, beta2, alpha, epsilon
    x = x0
    alpha = 0.1
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    # Initialization of momentum and adaptive terms
    m = np.zeros(x.shape[0])
    v = np.zeros(x.shape[0])

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        tic = time.time()
        k = iter + 1

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        g = gradf(x)
        
        # Compute adaptive parameters
        m_next = beta1 * m + (1 - beta1) * g
        v_next = beta2 * v + (1 - beta2) * (g ** 2)
        m_hat = m_next / (1 - (beta1 ** k))
        v_hat = v_next / (1 - (beta2 ** k))
        H = np.sqrt(v_hat) + epsilon

        # Compute the next x
        x_next = x - alpha * m_hat / H

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(
                iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        v = v_next
        m = m_next

    return x, info


def SGD(fx, gradf, parameter):
    """
    Function:  [x, info] = SGD(fx, gradf, parameter)
    Purpose:   Implementation of the stochastic gradient descent algorithm.
    Parameter: x0               - Initial estimate.
               maxit            - Maximum number of iterations.
               no0functions     - Number of datapoints
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    print(68*'*')
    print('Stochastic Gradient Descent')

    # Get parameters
    maxit = parameter['maxit']
    x0 = parameter['x0']
    s = parameter['no0functions']

    # Initialize x.
    x = x0

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        tic = time.time()
        k = iter + 1
        alpha = 1 / k

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables
        np.random.seed()
        i = np.random.randint(s - 1)
        x_next = x - alpha * gradf(x, i)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(
                iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    return x, info


def SAG(fx, gradf, parameter):
    """
    Function:  [x, info] = SAG(fx, gradf, parameter)
    Purpose:   Implementation of the stochastic gradient descent with averaging algorithm.
    Parameter: x0               - Initial estimate.
               maxit            - Maximum number of iterations.
               no0functions     - Number of datapoints
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    print(68*'*')
    print('Stochastic Gradient Descent with averaging')

    # Get parameters
    maxit = parameter['maxit']
    x0 = parameter['x0']
    size = parameter['no0functions']
    Lmax = parameter['Lmax']

    # Initialize x and alpha.
    x = x0
    v = np.zeros((size, x0.shape[0]))
    alpha = 1 / (16 * Lmax)

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}
    
    # Main loop.
    for iter in range(maxit):
        tic = time.time()
        k = iter + 1

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        np.random.seed()
        i = np.random.randint(size - 1)

        # Add to the averaging "history"
        v[i] = gradf(x, i)
        x_next = x - alpha / size * sum(v)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(
                iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    return x, info


def SVR(fx, gradf, gradfsto, parameter):
    """
    Function:  [x, info] = SVR(fx, gradf, gradfsto, parameter)
    Purpose:   Implementation of the stochastic gradient descent with variance reduction algorithm.
    Parameter: x0               - Initial estimate.
               maxit            - Maximum number of iterations.
               no0functions     - Number of datapoints
               Lmax             - Maximum Lipschitz constant
    :param fx:
    :param gradf:
    :param gradfsto:
    :param parameter:
    :return:
    """
    print(68*'*')
    print('Stochastic Gradient Descent with variance reduction')

    # Get parameters
    maxit = parameter['maxit']
    x0 = parameter['x0']
    size = parameter['no0functions']
    Lmax = parameter['Lmax']

    # Initialize x and alpha.
    gamma = 1e-2 / Lmax
    q = int(np.floor(1e3 * Lmax))
    x = x0

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.

    for iter in range(maxit):
        tic = time.time()
        np.random.seed()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        xt = x
        vk = gradf(xt)
        xtl = x
        xtl_list = []
        xtl_list.append(xtl)

        # Reduce variance
        for l in range(q - 1):
            i = np.random.randint(size - 1)
            vl = gradfsto(xtl, i) - gradfsto(xt, i) + vk
            xtl = xtl - gamma * vl
            xtl_list.append(xtl)

        # Get the next x
        x_next = 1 / q * np.sum(xtl_list, axis=0)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(
                iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    return x, info

def line_search(Lk_0, fx, x, d):
    """
    Function:   i = line_search(Lk_0, fx, x, d)
    Purpose:    Performs the line search to find the best i
    
    :param Lk_0:
    :param fx:
    :param x:
    :param d:
    """
    
    for i in count():
        # Factor that multiplies d_k
        factor_left = 1 / ((2 ** i) * Lk_0)
        # Factor that multiplies ||d_k||^2, adds the +1 power of 2 at the denominator on the right
        factor_right = factor_left / 2 
        # Left part of the inequality
        left = fx(x + factor_left * (-d))
        # Right part pf the inequality
        right = fx(x) - factor_right * np.linalg.norm(-d) ** 2
        if left <= right:
            return i