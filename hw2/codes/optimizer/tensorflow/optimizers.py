from abc import abstractmethod
import tensorflow as tf

class optimizer(object):
    @abstractmethod
    def update(self, gradients):
        raise NotImplementedError('compute_update function is not implemented.')

class SGDOptimizer(optimizer):
    def __init__(self, args):
        self.learning_rate = args.learning_rate

    def update(self, gradients):
       for i, g in enumerate(gradients):
           gradients[i] = tf.multiply(self.learning_rate, g)

class MomentumSGDOptimizer(optimizer):
    def __init__(self, args):
        self.learning_rate = args.learning_rate
        self.rho = args.rho
        self.m = None

    def update(self, gradients):
        if self.m is None:
            self.m = [tf.zeros_like(g) for g in gradients]

        for i, g in enumerate(gradients):
            self.m[i] = self.rho * self.m[i] + g
            gradients[i] = self.learning_rate * self.m[i]

class AdagradOptimizer(optimizer):
    def __init__(self, args):
        self.learning_rate = args.learning_rate
        self.delta = args.delta
        self.r = None

    def update(self, gradients):
        if self.r is None:
            self.r = [tf.zeros_like(g) for g in gradients]

        for i, g in enumerate(gradients):
            ## TODO.
            raise NotImplementedError('You should write your code HERE')


class RMSPropOptimizer(optimizer):
    def __init__(self, args):
        self.tau = args.tau
        self.learning_rate = args.learning_rate
        self.delta = args.delta
        self.r = None

    def update(self, gradients):
        if self.r is None:
            self.r = [tf.zeros_like(g) for g in gradients]

        for i,g in enumerate(gradients):
            ## TODO.
            raise NotImplementedError('You should write your code HERE')

class AdamOptimizer(optimizer):
    def __init__(self, args):
       self.beta1 = args.beta1
       self.beta2 = args.beta2
       self.learning_rate = args.learning_rate
       self.delta = args.delta
       self.iteration = None
       self.m1 = None
       self.m2 = None

    def update(self, gradients):
        if self.m1 is None:
            self.m1 = [tf.zeros_like(g) for g in gradients]
        if self.m2 is None:
            self.m2 = [tf.zeros_like(g) for g in gradients]
        if self.iteration is None:
            self.iteration = 1 

        for i, g in enumerate(gradients):
            ## TODO.
            raise NotImplementedError('You should write your code HERE')

        self.iteration = self.iteration + 1

def createOptimizer(args):
    if args.optimizer == "sgd":
        return SGDOptimizer(args)
    elif args.optimizer == "momentumsgd":
        return MomentumSGDOptimizer(args)
    elif args.optimizer == "adagrad":
        return AdagradOptimizer(args)
    elif args.optimizer == "rmsprop":
        return RMSPropOptimizer(args)
    elif args.optimizer == "adam":
        return AdamOptimizer(args)
