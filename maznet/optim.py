"""
We use an optimizer to adjust the parameters
of our network based on the gradients computed
during backpropagation
"""

from maznet.nn import NeuralNet


class Optimizer:
    """
    Base class for all optimizers
    """
    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic gradient descent
    """
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, net: NeuralNet) -> None:
        for param, grad in net.params_and_grads():
            param -= self.lr * grad
