from dataclasses import dataclass
from typing import Callable
from recurrent.mytypes import *
from torch.utils import _pytree as pytree


@dataclass(frozen=True, slots=True)
class RnnParameter(PYTREE):
    w_rec: PARAMETER
    w_out: PARAMETER
    n_h: int
    n_in: int
    n_out: int
    alpha: float
    activationFn: Callable[[torch.Tensor], torch.Tensor]


def rnnParameter_flatten(rnnParameter: RnnParameter):
    return (rnnParameter.w_rec, rnnParameter.w_out), (
        rnnParameter.n_h,
        rnnParameter.n_in,
        rnnParameter.n_out,
        rnnParameter.alpha,
        rnnParameter.activationFn,
    )


def rnnParameter_unflatten(children, aux):
    n_h, n_in, n_out, alpha, activationFn = aux
    return RnnParameter(
        w_rec=children[0],
        w_out=children[1],
        n_h=n_h,
        n_in=n_in,
        n_out=n_out,
        alpha=alpha,
        activationFn=activationFn,
    )


pytree.register_pytree_node(RnnParameter, rnnParameter_flatten, rnnParameter_unflatten)


@dataclass(frozen=True, slots=True)
class SgdParameter(PYTREE):
    learning_rate: LEARNING_RATE


def sgdParameter_flatten(sgdParameter: SgdParameter):
    return (sgdParameter.learning_rate,), None


def sgdParameter_unflatten(children, aux):
    return SgdParameter(learning_rate=children[0])


pytree.register_pytree_node(SgdParameter, sgdParameter_flatten, sgdParameter_unflatten)


@dataclass(frozen=True)
class RfloConfig:
    rflo_alpha: float
