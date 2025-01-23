from dataclasses import dataclass
from typing import Callable
from recurrent.mytypes import *
from torch.utils import _pytree as pytree
from torch import Tensor


class RnnParameter(NamedTuple):
    w_rec: PARAMETER
    w_out: PARAMETER


class SgdParameter(NamedTuple):
    learning_rate: LEARNING_RATE


class RfloConfig(NamedTuple):
    rflo_alpha: float


class RnnConfig(NamedTuple):
    n_h: int
    n_in: int
    n_out: int
    alpha: float
    activationFn: Callable[[torch.Tensor], torch.Tensor]


# can't make this into NamedTuple due to some bug where taking jacrev will lose the NamedTupleness
@dataclass(frozen=True, slots=True)
class UORO_Param[Pr]():
    A: Tensor
    B: Gradient[Pr]


def flatten[Pr](uoro: UORO_Param[Pr]):
    return (uoro.A, uoro.B), None


def unflatten[Pr](children, aux):
    return UORO_Param[Pr](*children)


pytree.register_pytree_node(UORO_Param, flatten, unflatten)


# def rnn_god_flatten[A, B, C](godState: RnnGodState[A, B, C]):
#     return (
#         godState.activation,
#         godState.influenceTensor,
#         godState.ohoInfluenceTensor,
#         godState.parameter,
#         godState.hyperparameter,
#         godState.metaHyperparameter,
#         godState.uoro,
#     ), (
#         godState.rfloConfig,
#         godState.rfloConfig_bilevel,
#         godState.rnnConfig,
#         godState.rnnConfig_bilevel,
#     )
