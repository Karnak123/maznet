"""
A NeuralNet is just a collecton of layers.
It behaves a lot like a leyers itself, although
we're not goint to make it one.
"""

from typing import Sequence
from tensor import Tensor
from layers import Layer


class NeuralNet:
    def __init__(self, layers: Sequence[Tensor]) -> None:
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
