from typing import Callable, Iterator, NamedTuple

from jaxtyping import PyTree

from .dataloader import NoneIterator


class Task(NamedTuple):
    model: PyTree
    loss_fn: Callable
    learning_data: Iterator = NoneIterator()
