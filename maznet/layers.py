"""
Our neural net will be made up of layers.
Each layer need to pass its inputs forward
and propagate gradients backwards.
"""

from typing import Dict, Callable
import numpy as np
from maznet.tensor import Tensor


class Layer:
    # A layer has a forward and backward pass
    def __init__(self) -> None:
        # Layer has no parameters
        self.params: Dict[str, Tensor] = dict()
        self.grads: Dict[str, Tensor] = dict()

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Produce the outputs corresponding to inputs
        """
        raise NotImplementedError

    def backwards(self, grad: Tensor) -> Tensor:
        """
        Backpropagate this gradient through the layer
        """
        raise NotImplementedError


class Linear(Layer):
    """
    Computes output = input @ w + b
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        # inputs will be (batch_size, input_size)
        # outputs will be (batch_size, output_size)
        super().__init__()
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        outputs = inputs @ w + b
        """
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backwards(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) and x = a * b + c
        then dy/da = f'(x) * b
        and dy/db = f'(x) * a
        and dy/dc = f'(x)

        if y = f(X) and a @ b + c
        then dy/da = f'(x) @ b.T
        and dy/db = a.T @ f'(x)
        and dy/dc = f'(x)
        """
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T


F = Callable[[Tensor], Tensor]


class Activation(Layer):
    """
    An activation layer just applies a function
    element-wise to its inputs
    """

    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backwards(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) and x = g(z)
        then dy/dz = f'(x) * g'(z)
        """
        return self.f_prime(self.inputs) * grad


def tanh(x: Tensor) -> Tensor:
    # return np.tanh(x)
    return np.tanh(x)


def tanh_prime(x: Tensor) -> Tensor:
    # return 1 - np.tanh(x) ** 2
    return 1 - tanh(x) ** 2


class Tanh(Activation):
    # Tanh is an activation function
    def __init__(self):
        super().__init__(tanh, tanh_prime)


def relu(x: Tensor) -> Tensor:
    # return np.maximum(0, x)
    return x * (x > 0)


def relu_prime(x: Tensor) -> Tensor:
    return (x > 0).astype(x.dtype)
    return 1 * (x > 0)


class ReLU(Activation):
    # ReLU is an activation function
    def __init__(self):
        super().__init__(relu, relu_prime)


def arctan(x: Tensor) -> Tensor:
    # return np.arctan(x)
    return np.arctan(x)


def arctan_prime(x: Tensor) -> Tensor:
    # return 1 / (1 + x ** 2)
    return 1 / (1 + x * x)


class ArcTan(Activation):
    # ArcTan is an activation function
    def __init__(self):
        super().__init__(arctan, arctan_prime)


def sigmoid(x: Tensor) -> Tensor:
    # return 1 / (1 + np.exp(-x))
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x: Tensor) -> Tensor:
    # return sigmoid(x) * (1 - sigmoid(x))
    return sigmoid(x) * (1 - sigmoid(x))


class Sigmoid(Activation):
    # Sigmoid is an activation function
    def __init__(self):
        super().__init__(sigmoid, sigmoid_prime)


def softplus(x: Tensor) -> Tensor:
    # return np.log(1 + np.exp(x))
    return np.log(1 + np.exp(x))


class SoftPlus(Activation):
    # SoftPlus is an activation function
    def __init__(self):
        super().__init__(softplus, sigmoid)
