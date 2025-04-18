from dataclasses import dataclass
from typing import Literal, Self
from donotation import do
import optax

from recurrent.monad import *
from recurrent.mytypes import *
from recurrent.parameters import *
from recurrent.util import prng_split


class InputOutput(eqx.Module):
    x: jax.Array
    y: jax.Array


class OhoData[Data](eqx.Module):
    payload: Data
    validation: Data


class GodState(eqx.Module):
    prng: PRNG
    start_epoch: int
    start_example: int
    innerTimeConstant: float = eqx.field(static=True)
    outerTimeConstant: float = eqx.field(static=True)
    globalLogConfig: GlobalLogConfig = eqx.field(static=True)

    # Inner fields
    rnnState: Optional[RnnState] = eqx.field(default=None)
    innerInfluenceTensor: Optional[JACOBIAN] = eqx.field(default=None)
    innerUoro: Optional[UORO_Param] = eqx.field(default=None)
    innerLogs: Optional[Logs] = eqx.field(default=None)
    innerLogConfig: Optional[LogConfig] = eqx.field(default=None, static=True)
    innerOptState: Optional[optax.OptState] = eqx.field(default=None)
    innerSgdParameter: Optional[SgdParameter] = eqx.field(default=None)
    innerAdamParameter: Optional[AdamParameter] = eqx.field(default=None)

    # Outer fields
    outerInfluenceTensor: Optional[JACOBIAN] = eqx.field(default=None)
    outerUoro: Optional[UORO_Param] = eqx.field(default=None)
    outerLogs: Optional[Logs] = eqx.field(default=None)
    outerLogConfig: Optional[LogConfig] = eqx.field(default=None, static=True)
    outerOptState: Optional[optax.OptState] = eqx.field(default=None)
    outerSgdParameter: Optional[SgdParameter] = eqx.field(default=None)
    outerAdamParameter: Optional[AdamParameter] = eqx.field(default=None)


@dataclass(frozen=True)
class GodInterpreter:
    type LocalApp[X] = App[Self, GodState, X]

    getRecurrentState: LocalApp[REC_STATE]
    putRecurrentState: Callable[[REC_STATE], LocalApp[Unit]]
    getRecurrentParam: LocalApp[REC_PARAM]
    putRecurrentParam: Callable[[REC_PARAM], LocalApp[Unit]]

    getActivation: LocalApp[ACTIVATION]
    putActivation: Callable[[ACTIVATION], LocalApp[Unit]]
    getInfluenceTensor: LocalApp[JACOBIAN]
    putInfluenceTensor: Callable[[JACOBIAN], LocalApp[Unit]]
    getUoro: LocalApp[UORO_Param]
    putUoro: Callable[[UORO_Param], LocalApp[Unit]]
    getRnnConfig: LocalApp[RnnConfig]
    getTimeConstant: LocalApp[float]
    getLogConfig: LocalApp[LogConfig]
    getGlobalLogConfig: LocalApp[GlobalLogConfig]
    putLogs: Callable[[Logs], LocalApp[Unit]]

    getRnnParameter: LocalApp[RnnParameter]
    putRnnParameter: Callable[[RnnParameter], LocalApp[Unit]]

    getOptState: LocalApp[optax.OptState]
    putOptState: Callable[[optax.OptState], LocalApp[Unit]]
    getOptimizer: LocalApp[optax.GradientTransformation]
    getUpdater: LocalApp[Callable[[optax.Params, optax.Updates], optax.Params]]
    getLearningRate: Callable[[GodState], float]

    @do()
    def updatePRNG(self) -> G[LocalApp[PRNG]]:
        prng, new_prng = yield from gets(lambda e: prng_split(e.prng))
        _ = yield from modifies(lambda e: eqx.tree_at(lambda t: t.prng, e, new_prng))
        return pure(prng, PX[tuple[Self, GodState]]())


@dataclass(frozen=True)
class GodConfig:
    data_load_size: int
    num_retrain_loops: int
    checkpoint_interval: int
    inner_learning_rate: float
    outer_learning_rate: float
    ts: tuple[int, int]
    data_seed: int
    parameter_seed: int
    test_seed: int
    tr_examples_per_epoch: int
    vl_examples_per_epoch: int
    tr_avg_per: int
    numVal: int
    numTr: int
    numTe: int
    inner_learner: Literal["rtrl", "uoro", "rflo", "identity"]
    outer_learner: Literal["rtrl", "uoro", "rflo", "identity"]
    lossFn: Literal["cross_entropy"]
    inner_optimizer: Literal["sgd", "sgd_positive", "adam", "sgd_normalized", "sgd_clipped"]
    outer_optimizer: Literal["sgd", "sgd_positive", "adam", "sgd_normalized", "sgd_clipped"]
    inner_optimizer_parametrization: Literal["identity", "softplus"]
    outer_optimizer_parametrization: Literal["identity", "softplus"]
    activation_fn: Literal["tanh", "relu"]
    architecture: Literal["rnn"]
    n_h: int
    n_in: int
    n_out: int
    inner_time_constant: float
    outer_time_constant: float
    tau_task: bool
    inner_uoro_std: float
    outer_uoro_std: float
    initialization_std: float
    inner_log_special: bool
    outer_log_special: bool
    inner_lanczos_iterations: int
    outer_lanczos_iterations: int
    inner_clip: float
    inner_clip_sharpness: float
    outer_clip: float
    outer_clip_sharpness: float
    inner_log_expensive: Optional[bool] = None
    outer_log_expensive: Optional[bool] = None
