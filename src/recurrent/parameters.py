from typing import Literal, Optional

from recurrent.mytypes import *
import jax
import equinox as eqx


class SgdParameter(eqx.Module):
    learning_rate: jax.Array


class AdamParameter(eqx.Module):
    learning_rate: jax.Array


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
    log_influence: bool = eqx.field(static=True, default=False)
    log_accumulate_influence: bool = eqx.field(static=True, default=False)


class AllLogs(eqx.Module):
    train_loss: jax.Array | None
    validation_loss: jax.Array | None
    test_loss: jax.Array | None
    meta_test_loss: jax.Array | None
    hyperparameters: jax.Array | None
    inner_learning_rate: jax.Array | None
    parameter_norm: jax.Array | None
    oho_gradient: jax.Array | None
    train_gradient: jax.Array | None
    validation_gradient: jax.Array | None
    immediate_influence_tensor_norm: jax.Array | None
    outer_influence_tensor_norm: jax.Array | None
    outer_influence_tensor: jax.Array | None
    inner_influence_tensor_norm: jax.Array | None
    largest_jacobian_eigenvalue: jax.Array | None
    largest_hessian_eigenvalue: jax.Array | None
    jacobian: jax.Array | None
    hessian: jax.Array | None
    rnn_activation_norm: jax.Array | None
    immediate_influence_tensor: jax.Array | None


class CustomSequential(eqx.Module):
    model: eqx.nn.Sequential

    def __init__(self, layer_defs: list[tuple[int, Callable[[jax.Array], jax.Array]]], input_size: int, key: PRNG):
        layers = []
        in_size = input_size
        layer_keys = jax.random.split(key, len(layer_defs))

        for (out_size, activation), k in zip(layer_defs, layer_keys):
            layers.append(eqx.nn.Linear(in_size, out_size, key=k))
            layers.append(eqx.nn.Lambda(activation))
            in_size = out_size

        self.model = eqx.nn.Sequential(layers)

    def __call__(self, x):
        return self.model(x)
