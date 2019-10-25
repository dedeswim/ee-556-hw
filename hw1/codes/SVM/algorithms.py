import time
import numpy as np
import scipy.sparse.linalg as spla
from numpy.random import randint
from scipy.sparse.linalg.dsolve import linsolve
from itertools import count

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
        if (iter %  5 ==0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
    info['iter'] = maxit
    return x, info


# gradient with strong convexity
def GDstr(fx, gradf, parameter) :
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
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    return x, info

# accelerated gradient
def AGD(fx, gradf, parameter):
    """
    *******************  EE556 - Mathematics of Data  ************************
    Function:  AGD (fx, gradf, parameter)
    Purpose:   Implementation of the accelerated gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
                 strcnvx	- strong convexity parameter
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

    # Initialize x, y and t.
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
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))


        # Prepare next iteration
        x = x_next
        t = t_next

    return x, info

# accelerated gradient with strong convexity
def AGDstr(fx, gradf, parameter):
    """
    *******************  EE556 - Mathematics of Data  ************************
    Function:  AGDstr(fx, gradf, parameter)
    Purpose:   Implementation of the accelerated gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
                strcnvx	- strong convexity parameter
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

    # Initialize x, y and gamma.
    alpha = 1 / Lips
    gamma = (np.math.sqrt(Lips) - np.math.sqrt(strcnvx)) / (np.math.sqrt(Lips) + np.math.sqrt(strcnvx))
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
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))


        # Prepare next iteration
        x = x_next

    return x, info

# LSGD
def LSGD(fx, gradf, parameter):
    """
    Function:  [x, info] = LSGD(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent with line-search.
    Parameter: x0         - Initial estimate.
           maxit      - Maximum number of iterations.
           Lips       - Lipschitz constant for gradient.
           strcnvx    - Strong convexity parameter of f(x).
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

        for i in count():
            factor_left = 1 / ((2 ** i) * Lk_0)
            factor_right = factor_left / 2 # Adds the +1 power at the denominator on th right
            left = fx(x + factor_left * (-d))
            right = fx(x) - factor_right * np.linalg.norm(d) ** 2          
            if left <= right:
                break
        
        Lk_0 = (2 ** i) * Lk_0       
        x_next = x - 1 / Lk_0 * d

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare next iteration
        x = x_next

    return x, info

# LSAGD
def LSAGD(fx, gradf, parameter):
    """
    Function:  [x, info] = LSAGD (fx, gradf, parameter)
    Purpose:   Implementation of AGD with line search.
    Parameter: x0         - Initial estimate.
           maxit      - Maximum number of iterations.
           Lips       - Lipschitz constant for gradient.
           strcnvx    - Strong convexity parameter of f(x).
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
        
        for i in count():
            factor_left = 1 / ((2 ** i) * Lk_0)
            factor_right = factor_left / 2 # Adds the +1 power at the denominator on th right
            left = fx(y + factor_left * (-d))
            right = fx(y) - factor_right * np.linalg.norm(-d) ** 2          
            if left <= right:
                break
        
        L_k = (2 ** i) * Lk_0
        t_next =  0.5 * (1 + np.math.sqrt(1 + 4 * (L_k / Lk_old) * (t ** 2))) / 2
        x_next = y - 1 / L_k * d
        y = x_next + (t - 1) / t_next * (x_next - x)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

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
    Parameter: x0         - Initial estimate.
           maxit      - Maximum number of iterations.
           Lips       - Lipschitz constant for gradient.
           strcnvx    - Strong convexity parameter of f(x).
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
        fval_next = fx(x_next)
        
        if fval_next > fval:
            y = x
            t = 1
            x_next = y - alpha * gradf(y)
            fval_next = fx(x_next)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        t = t_next
        fval = fval_next

    return x, info


# LSAGDR
def LSAGDR(fx, gradf, parameter):
    """
    Function:  [x, info] = LSAGDR (fx, gradf, parameter)
    Purpose:   Implementation of AGD with line search and adaptive restart.
    Parameter: x0         - Initial estimate.
           maxit      - Maximum number of iterations.
           Lips       - Lipschitz constant for gradient.
           strcnvx    - Strong convexity parameter of f(x).
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
        
        for i in count():
            factor_left = 1 / ((2 ** i) * Lk_0)
            factor_right = factor_left / 2 # Adds the +1 power at the denominator on th right
            left = fx(y + factor_left * (-d))
            right = fx(y) - factor_right * np.linalg.norm(-d) ** 2          
            if left <= right:
                break
        
        L_k = (2 ** i) * Lk_0
        t_next =  0.5 * (1 + np.math.sqrt(1 + 4 * (L_k / Lk_old) * (t ** 2))) / 2
        x_next = y - 1 / L_k * d
        y = x_next + (t - 1) / t_next * (x_next - x)
        fval_next = fx(x_next)

        if fval_next > fval:
            y = x
            t = 1
            d = gradf(y)
            for i in count():
                factor_left = 1 / ((2 ** i) * Lk_0)
                factor_right = factor_left / 2 # Adds the +1 power at the denominator on th right
                left = fx(y + factor_left * (-d))
                right = fx(y) - factor_right * np.linalg.norm(-d) ** 2          
                if left <= right:
                    break
            
            L_k = (2 ** i) * Lk_0
            t_next =  0.5 * (1 + np.math.sqrt(1 + 4 * (L_k / Lk_old) * (t ** 2))) / 2
            x_next = y - 1 / L_k * d
            y = x_next + (t - 1) / t_next * (x_next - x)
            fval_next = fx(x_next)



        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        t = t_next
        Lk_0 = L_k
        fval = fval_next

    return x, info

def AdaGrad(fx, gradf, parameter):
    """
    Function:  [x, info] = AdaGrad (fx, gradf, hessf, parameter)
    Purpose:   Implementation of the adaptive gradient method with scalar step-size.
    Parameter: x0         - Initial estimate.
           maxit      - Maximum number of iterations.
           Lips       - Lipschitz constant for gradient.
           strcnvx    - Strong convexity parameter of f(x).
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
        Q_next = Q + np.linalg.norm(grad) ** 2
        H_k = np.math.sqrt(Q_next + delta)
        x_next = x - alpha * np.dot((1 / H_k) * np.eye(grad.shape[0]), grad)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.clock() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        Q = Q_next

    return x, info

# Newton
def ADAM(fx, gradf, parameter):
    """
    Function:  [x, info] = ADAM (fx, gradf, hessf, parameter)
    Purpose:   Implementation of ADAM.
    Parameter: x0         - Initial estimate.
           maxit      - Maximum number of iterations.
           Lips       - Lipschitz constant for gradient.
           strcnvx    - Strong convexity parameter of f(x).
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

    # Initialize x, beta1, beta2, alpha, epsilon (and any other)
    x = x0
    alpha = 0.1
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    # Initialization of momentum and adaptive term
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
        m_next = beta1 * m + (1 - beta1) * g
        v_next = beta2 * v + (1 - beta2) * (g ** 2)
        m_hat = m_next / (1 - (beta1 ** k))
        v_hat = v_next / (1 - (beta2 ** k))
        H = np.sqrt(v_hat) + epsilon

        x_next = x - alpha * m_hat / H

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        v = v_next
        m = m_next

    return x, info

def SGD(fx, gradf, parameter):
    """
    Function:  [x, info] = GD(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
               strcnvx    - Strong convexity parameter of f(x).
               size       - Size of the dataset (needed to give a max to the random integer)
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
    size = parameter['no0functions']

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
        i = np.random.randint(size - 1)
        x_next = x - alpha * gradf(x, i)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    return x, info


def SAG(fx, gradf, parameter):
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
    print('Stochastic Gradient Descent with averaging')

    # Initialize x and alpha.
    #### YOUR CODES HERE


    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}
    # Main loop.

    for iter in range(maxit):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        
        #### YOUR CODES HERE

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    return x, info

def SVR(fx, gradf, gradfsto, parameter):
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
    print('Stochastic Gradient Descent with variance reduction')

    # Initialize x and alpha.
    #### YOUR CODES HERE

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.

    for iter in range(maxit):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        
        #### YOUR CODES HERE

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x= x_next

    return x, info



