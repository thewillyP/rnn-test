from typing import Generator, Generic, NamedTuple, Protocol, TypeVar, NewType
import torch
from torch.utils import _pytree as pytree


type G[T] = Generator[None, None, T]


class PYTREE(Protocol):
    pass


ACTIVATION = NewType("ACTIVATION", torch.Tensor)
PREDICTION = NewType("PREDICTION", torch.Tensor)
PARAMETER = NewType("PARAMETER", torch.Tensor)
# INFLUENCETENSOR = NewType("INFLUENCETENSOR", torch.Tensor)
# HYPERPARAMETER = NewType("HYPERPARAMETER", torch.Tensor)
LOSS = NewType("LOSS", torch.Tensor)  # float32
LEARNING_RATE = NewType("LEARNING_RATE", torch.Tensor)


# bc grad over param pytree returns a param pytree, and need a way to distinguish between gr vs param, but still recognize they're same type
class Gradient[T: PYTREE](NamedTuple):
    value: T


def gradient_flatten[T](child: Gradient[T]):
    return (child.value,), None


def gradient_unflatten[T](children: tuple[T], _):
    (value,) = children
    return Gradient[T](value=value)


pytree.register_pytree_node(Gradient, gradient_flatten, gradient_unflatten)  # type: ignore
