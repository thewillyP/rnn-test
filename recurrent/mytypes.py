from typing import Callable, Generator, NewType
import jax
import equinox as eqx
from equinox import Module

type G[T] = Generator[None, None, T]


# bc grad over param pytree returns a param pytree, and need a way to distinguish between gr vs param, but still recognize they're same type
class Gradient[T: Module | jax.Array](Module):
    value: jax.Array

    def __add__(self, other):
        if isinstance(other, Gradient):
            return Gradient(self.value + other.value)
        raise TypeError("Unsupported operand type(s) for +: 'Gradient' and '{}'".format(type(other).__name__))


class Jacobian[T: Module | jax.Array](Module):
    value: jax.Array

    def __add__(self, other):
        if isinstance(other, Jacobian):
            return Jacobian[T](self.value + other.value)
        raise TypeError("Unsupported operand type(s) for +: 'Jacobian' and '{}'".format(type(other).__name__))


ACTIVATION = NewType("ACTIVATION", jax.Array)
PREDICTION = NewType("PREDICTION", jax.Array)
LOSS = NewType("LOSS", jax.Array)  # is a scalar
PRNG = NewType("PRNG", jax.Array)


# smart constructor to let me know that some PyTrees are meant to be iterated
class Traversable[T: Module](Module):
    value: T | jax.Array


# class VLeaf[T](eqx.Module):
#     vector: jax.Array
#     toParam: Callable[[jax.Array], T] = eqx.field(static=True)


# class VNode[T](eqx.Module):
#     vector: T


# type IsVector[T] = VLeaf[T] | VNode[IsVector[T]]


class IsVector[T](Module):
    vector: jax.Array
    toParam: Callable[[jax.Array], T] = eqx.field(static=True)


def invmap[T](isVector: IsVector[T], f: Callable[[jax.Array], jax.Array]) -> IsVector[T]:
    vs, _ = jax.tree.flatten(isVector)
    return eqx.tree_at(lambda t: t.vector, isVector, f(vs[0]))


def endowVector[T: Module](tree: T) -> IsVector[T]:
    vector, toParam = jax.flatten_util.ravel_pytree(tree)
    return IsVector(vector=vector, toParam=toParam)


def toVector[T: Module](isVector: IsVector[T]) -> jax.Array:
    vs, _ = jax.tree.flatten(isVector)
    return vs[0]


def toParam[T: Module](isVector: IsVector[T]) -> T:
    vs, _ = jax.tree.flatten(isVector)
    return isVector.toParam(vs[0])


def identityVector(vector: jax.Array) -> IsVector[jax.Array]:
    return IsVector(vector=vector, toParam=lambda x: x)
