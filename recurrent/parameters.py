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


class UORO_Param[Pr](NamedTuple):
    A: Tensor
    B: Gradient[Pr]
