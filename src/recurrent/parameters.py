from typing import Literal, Optional

from recurrent.mytypes import *
import jax
import equinox as eqx


class SgdParameter(eqx.Module):
    learning_rate: float


class AdamParameter(eqx.Module):
    learning_rate: float


class RnnParameter(eqx.Module):
    w_rec: jax.Array
    w_out: jax.Array


class RnnConfig(eqx.Module):
    n_h: int
    n_in: int
    n_out: int
    activationFn: Literal["tanh", "relu"]


class RnnState(eqx.Module):
    activation: ACTIVATION
    rnnParameter: RnnParameter
    rnnConfig: RnnConfig = eqx.field(static=True)


class UORO_Param(eqx.Module):
    A: jax.Array
    B: jax.Array


class Logs(eqx.Module):
    gradient: Optional[jax.Array] = eqx.field(default=None)
    validationGradient: Optional[jax.Array] = eqx.field(default=None)
    influenceTensor: Optional[jax.Array] = eqx.field(default=None)
    immediateInfluenceTensor: Optional[jax.Array] = eqx.field(default=None)
    jac_eigenvalue: Optional[jax.Array] = eqx.field(default=None)
    hessian: Optional[jax.Array] = eqx.field(default=None)


class LogConfig(eqx.Module):
    log_special: bool = eqx.field(static=True)
    lanczos_iterations: int = eqx.field(static=True)
    log_expensive: bool = eqx.field(static=True)


class GlobalLogConfig(eqx.Module):
    stop_influence: bool = eqx.field(static=True, default=False)
