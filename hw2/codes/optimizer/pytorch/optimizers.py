from abc import abstractmethod
import torch


class optimizer(object):
    @abstractmethod
    def update(self, gradients):
        raise NotImplementedError('update function is not implemented.')


class SGDOptimizer(optimizer):
    def __init__(self, args):
        self.learning_rate = args.learning_rate

    def update(self, model):
        for p in model.parameters():
            p.grad *= self.learning_rate


class MomentumSGDOptimizer(optimizer):
    def __init__(self, args):
        self.learning_rate = args.learning_rate
        self.rho = args.rho
        self.m = None

    def update(self, model):
        if self.m is None:
            self.m = [torch.zeros(p.size()) for p in model.parameters()]

        for i, p in enumerate(model.parameters()):
            self.m[i] = self.rho * self.m[i] + p.grad
            p.grad = self.learning_rate * self.m[i]


class AdagradOptimizer(optimizer):
    def __init__(self, args):
        self.delta = args.delta
        self.learning_rate = args.learning_rate
        self.r = None

    def update(self, model):
        if self.r is None:
            self.r = [torch.zeros(p.size()) for p in model.parameters()]

        for i, p in enumerate(model.parameters()):
            self.r[i] += p.grad ** 2
            p.grad *= self.learning_rate / (self.delta + torch.sqrt(self.r[i]))


class RMSPropOptimizer(optimizer):
    def __init__(self, args):
        self.tau = args.tau
        self.learning_rate = args.learning_rate
        self.r = None
        self.delta = args.delta

    def update(self, model):
        if self.r is None:
            self.r = [torch.zeros(p.size()) for p in model.parameters()]

        for i, p in enumerate(model.parameters()):
            self.r[i] = self.tau * self.r[i] + (1 - self.tau) * p.grad ** 2
            p.grad *= self.learning_rate / (self.delta + torch.sqrt(self.r[i]))


class AdamOptimizer(optimizer):
    def __init__(self, args):
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.learning_rate = args.learning_rate
        self.delta = args.delta
        self.iteration = None
        self.m1 = None
        self.m2 = None

    def update(self, model):
        if self.m1 is None:
            self.m1 = [torch.zeros(p.grad.size()) for p in model.parameters()]
        if self.m2 is None:
            self.m2 = [torch.zeros(p.grad.size()) for p in model.parameters()]
        if self.iteration is None:
            self.iteration = 1

        for i, p in enumerate(model.parameters()):
            
            self.m1[i] = self.beta1 * self.m1[i] + (1 - self.beta1) * p.grad
            self.m2[i] = self.beta2 * self.m2[i] + (1 - self.beta2) * p.grad ** 2

            m1_hat = self.m1[i] / (1 - self.beta1 ** (self.iteration + 1))
            m2_hat = self.m2[i] / (1 - self.beta2 ** (self.iteration + 1))
            p.grad = self.learning_rate * m1_hat / (self.delta + m2_hat.sqrt())

        self.iteration = self.iteration+1


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
