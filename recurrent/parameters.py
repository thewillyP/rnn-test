from typing import Callable, Optional
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
