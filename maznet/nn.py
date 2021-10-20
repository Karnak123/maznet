"""
A NeuralNet is just a collecton of layers.
It behaves a lot like a leyers itself, although
we're not goint to make it one.
"""

from typing import Sequence, Iterator, Tuple
from maznet.tensor import Tensor
from maznet.layers import Layer


class NeuralNet:
    """
    A NeuralNet is just a collection of layers.
    It behaves a lot like a leyers itself, although
    we're not goint to make it one.
    """
    def __init__(self, layers: Sequence[Tensor]) -> None:
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor:
        # Forward pass through all layers
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad: Tensor) -> Tensor:
        # Backward pass through all layers
        for layer in reversed(self.layers):
            grad = layer.backwards(grad)
        return grad

    def params_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:
        # Generator for iterating over all parameters and gradients
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad
