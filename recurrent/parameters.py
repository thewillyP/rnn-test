from typing import Callable, Optional

import jax.flatten_util
from recurrent.mytypes import *
import jax
import equinox as eqx


class RnnParameter(eqx.Module):
    w_rec: jax.Array
    w_out: jax.Array


class SgdParameter(eqx.Module):
    learning_rate: jax.Array


class RnnConfig(eqx.Module):
    n_h: int
    n_in: int
    n_out: int
    alpha: float
    activationFn: Callable[[jax.Array], jax.Array]


class UORO_Param(eqx.Module):
    A: jax.Array
    B: jax.Array


class Logs(eqx.Module):
    train_loss: Optional[LOSS] = eqx.field(default=None)
    validation_loss: Optional[LOSS] = eqx.field(default=None)
    test_loss: Optional[LOSS] = eqx.field(default=None)
    learning_rate: Optional[jax.Array] = eqx.field(default=None)
    effective_learning_rate: Optional[jax.Array] = eqx.field(default=None)
