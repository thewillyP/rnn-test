from typing import Callable, Optional

import jax.flatten_util
from recurrent.mytypes import *
import jax
import equinox as eqx


class RnnParameter(eqx.Module):
    w_rec: PARAMETER
    w_out: PARAMETER


class SgdParameter(eqx.Module):
    learning_rate: LEARNING_RATE


class RfloConfig(eqx.Module):
    rflo_alpha: float


class RnnConfig(eqx.Module):
    n_h: int
    n_in: int
    n_out: int
    alpha: float
    activationFn: Callable[[jax.Array], jax.Array]


class UORO_Param[Pr: eqx.Module](eqx.Module):
    A: jax.Array
    B: Gradient[Pr]


class Logs(eqx.Module):
    loss: Optional[LOSS] = eqx.field(default=None)


class IsVector[T: eqx.Module](eqx.Module):
    vector: jax.Array
    toParam: Callable[[jax.Array], T] = eqx.field(static=True)


def endowVector(tree: eqx.Module) -> IsVector[eqx.Module]:
    vector, toParam = jax.flatten_util.ravel_pytree(tree)
    return IsVector(vector=vector, toParam=toParam)


def toVector[T: eqx.Module](isVector: IsVector[T]) -> jax.Array:
    return isVector.vector


def toParam[T: eqx.Module](isVector: IsVector[T]) -> T:
    return isVector.toParam(isVector.vector)


def updateVector[
    T: eqx.Module
](isVector: IsVector[T], vector: jax.Array) -> IsVector[T]:
    return IsVector(vector=vector, toParam=isVector.toParam)
