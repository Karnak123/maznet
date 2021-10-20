"""
A loss function measures how good our predictions are,
we use this to adjust the parameters of out network
"""

import numpy as np
from maznet.tensor import Tensor


class Loss:
    # A loss function measures how good our predictions are,
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError


class MSE(Loss):
    """
    MSE is mean squared error
    mse = (A - B)^2/n
    """

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.mean((predicted - actual) ** 2)

    def grad(self, predicted: Tensor, actual: Tensor) -> float:
        return (predicted - actual) * (2 / len(predicted))


class MAE(Loss):
    """
    MAE is mean absolute error
    mae = |A - B|/n
    """

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.mean(np.abs(predicted - actual))

    def grad(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sign(actual - predicted) / predicted.size


class SAE(Loss):
    """
    SAE is sum absolute error
    sae = |A - B|
    """

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum(np.abs(predicted - actual))

    def grad(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sign(actual - predicted)


class CCE(Loss):
    """
    CCE is cross entropy error
    cce = -sum( t * log(o) + (1 - t) * log(1 - o))
    """

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        y = predicted.copy()
        t = actual.copy()
        eps = np.spacing(1)
        y[y > (1 - eps)] = 1 - eps
        y[y < eps] = eps
        t[t > (1 - eps)] = 1 - eps
        t[t < eps] = eps
        return -np.sum(t * np.log(y) + (1 - t) * np.log(1 - y)) / t.size

    def grad(self, predicted: Tensor, actual: Tensor) -> float:
        y = predicted.copy()
        t = actual.copy()
        eps = 0.0
        y[y > (1 - eps)] = 1 - eps
        y[y < eps] = eps
        t[t > (1 - eps)] = 1 - eps
        t[t < eps] = eps
        # dC/dy = - d/y + (1-d)/(1-y)
        eps = np.spacing(1)
        return (t / (y + eps) - (1 - t) / (1 - y + eps)) / t.size
