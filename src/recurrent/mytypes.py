from typing import Callable, Generator, NewType, Protocol
import jax
import equinox as eqx
from equinox import Module

type G[T] = Generator[None, None, T]
type CanDiff = jax.Array | Module


# bc grad over param pytree returns a param pytree, and need a way to distinguish between gr vs param, but still recognize they're same type
class Gradient[T: CanDiff](Module):
    value: jax.Array

    def __add__(self, other):
        if isinstance(other, Gradient):
            return Gradient(self.value + other.value)
        raise TypeError("Unsupported operand type(s) for +: 'Gradient' and '{}'".format(type(other).__name__))


class Jacobian[T: CanDiff](Module):
    value: jax.Array

    def __add__(self, other):
        if isinstance(other, Jacobian):
            return Jacobian[T](self.value + other.value)
        raise TypeError("Unsupported operand type(s) for +: 'Jacobian' and '{}'".format(type(other).__name__))


ACTIVATION = NewType("ACTIVATION", jax.Array)
PREDICTION = NewType("PREDICTION", jax.Array)
GRADIENT = NewType("GRADIENT", jax.Array)  # is a vector
JACOBIAN = NewType("JACOBIAN", jax.Array)  # is a matrix

INPUT = NewType("INPUT", jax.Array)  # is a vector
LABEL = NewType("LABEL", jax.Array)  # is a vector
PREDICTION_INPUT = NewType("PREDICTION_INPUT", jax.Array)  # is a vector

REC_STATE = NewType("REC_STATE", jax.Array)  # is a vector
REC_PARAM = NewType("REC_PARAM", jax.Array)  # is a vector

LOSS = NewType("LOSS", jax.Array)  # is a scalar
PRNG = NewType("PRNG", jax.Array)


class Wrap[T](Protocol[T]):
    value: T | jax.Array


class IdentityF[T](Module):
    value: T | jax.Array


# smart constructor to let me know that some PyTrees are meant to be iterated
class Traversable[T: Module](Module):
    value: T | jax.Array


class IsVector[T](Module):
    vector: jax.Array
    nonparams: T
    toParam: Callable[[jax.Array], T] = eqx.field(static=True)


def invmap[T: CanDiff](canVec: T, f: Callable[[jax.Array], jax.Array]):
    vectorized = endowVector(canVec)
    new_vector = f(vectorized.vector)
    return eqx.combine(vectorized.toParam(new_vector), vectorized.nonparams)


def endowVector[T: CanDiff](tree: T) -> IsVector[T]:
    params, nonparams = eqx.partition(tree, eqx.is_array)
    vector, toParam = jax.flatten_util.ravel_pytree(params)
    return IsVector(vector=vector, nonparams=nonparams, toParam=toParam)


def toVector[T: CanDiff](isVector: IsVector[T]) -> jax.Array:
    return isVector.vector


def toParam[T: CanDiff](isVector: IsVector[T]) -> T:
    return eqx.combine(isVector.toParam(isVector.vector), isVector.nonparams)
