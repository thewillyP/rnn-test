from dataclasses import dataclass
from recurrent.mytypes import *
from torch.utils import _pytree as pytree


@dataclass(frozen=True, slots=True)
class RnnParameter:
    w_rec: PARAMETER
    w_out: PARAMETER


def rnnParameter_flatten(rnnParameter: RnnParameter):
    return (rnnParameter.w_rec, rnnParameter.w_out), None


def rnnParameter_unflatten(children, aux):
    return RnnParameter(w_rec=children[0], w_out=children[1])


pytree.register_pytree_node(RnnParameter, rnnParameter_flatten, rnnParameter_unflatten)


@dataclass(frozen=True, slots=True)
class SgdParameter:
    learning_rate: LEARNING_RATE


def sgdParameter_flatten(sgdParameter: SgdParameter):
    return (sgdParameter.learning_rate,), None


def sgdParameter_unflatten(children, aux):
    return SgdParameter(learning_rate=children[0])


pytree.register_pytree_node(SgdParameter, sgdParameter_flatten, sgdParameter_unflatten)


@dataclass(frozen=True)
class RfloConfig:
    rflo_alpha: float
