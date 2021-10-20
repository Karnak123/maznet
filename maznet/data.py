"""
We'll feed inputs into our network in batches.
So here are some tools for iterating over our data in batches.
"""

from typing import Iterator, NamedTuple
import numpy as np
from maznet.tensor import Tensor

Batch = NamedTuple("Batch", [("inputs", Tensor), ("targets", Tensor)])


class DataIterator:
    # Iterate over batches of data
    def __call__(self, inputs: Tensor, targets: Tensor) -> Tensor:
        raise NotImplementedError


class BatchIterator(DataIterator):
    # Iterate over batches of data
    def __init__(self, batch_size: int = 32, shuffle: bool = True) -> None:
        # batch_size: How many examples in each batch
        # shuffle: Whether to shuffle the data before each epoch
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs: Tensor, targets: Tensor) -> Tensor:
        # inputs: Tensor of shape (N, C, H, W)
        # targets: Tensor of shape (N,)
        starts = np.arange(0, len(inputs), self.batch_size)
        if self.shuffle:
            np.random.shuffle(starts)

        for start in starts:
            end = start + self.batch_size
            batch_inputs = inputs[start:end]
            batch_targets = targets[start:end]
            yield Batch(batch_inputs, batch_targets)
