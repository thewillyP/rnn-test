from typing import Callable, Optional

import jax.flatten_util
from recurrent.mytypes import *
import jax
import equinox as eqx
import optax


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
    activationFn: Callable[[jax.Array], jax.Array]


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


class AllLogs(eqx.Module):
    trainLoss: jax.Array
    validationLoss: jax.Array
    testLoss: jax.Array
    hyperparameters: jax.Array
    parameterNorm: jax.Array
    ohoGradient: jax.Array
    trainGradient: jax.Array
    validationGradient: jax.Array
    immediateInfluenceTensorNorm: jax.Array
    outerInfluenceTensorNorm: jax.Array
    innerInfluenceTensorNorm: jax.Array
    largest_hessian_eigenvalue: jax.Array
    largest_jacobian_eigenvalue: jax.Array


class LogConfig(eqx.Module):
    log_special: bool = eqx.field(static=True)
    lanczos_iterations: int = eqx.field(static=True)
