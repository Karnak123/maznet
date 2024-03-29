"""
Here's a function that can train a neural network
"""

from maznet.tensor import Tensor
from maznet.nn import NeuralNet
from maznet.loss import Loss, MSE
from maznet.optim import Optimizer, SGD
from maznet.data import DataIterator, BatchIterator


def train(
    net: NeuralNet,
    inputs: Tensor,
    targets: Tensor,
    num_epochs: int = 5000,
    iterator: DataIterator = BatchIterator(),
    loss: Loss = MSE(),
    optimizer: Optimizer = SGD(),
) -> None:
    # Trains a neural network using mini-batch gradient descent.
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            predicted = net.forward(batch.inputs)
            epoch_loss += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets)
            net.backward(grad)
            optimizer.step(net)
        print(epoch, epoch_loss)
