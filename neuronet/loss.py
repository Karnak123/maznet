"""
A loss function measures how good our predictions are,
we use this to adjust the parameters of out network
"""

import numpy as np
from neuronet.tensor import Tensor


class Loss:
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
