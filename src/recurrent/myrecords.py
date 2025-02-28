from dataclasses import dataclass
from typing import Self
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


@dataclass(frozen=True)
class GodState(eqx.Module):
    prng: PRNG
    logConfig: LogConfig = eqx.field(static=True)
    lossFn: Callable[[jax.Array, jax.Array], jax.Array] = eqx.field(static=True)

    # inner
    rnnState: RnnState
    innerInfluenceTensor: JACOBIAN
    innerUoro: UORO_Param
    innerLogs: Logs
    innerTimeConstant: float = eqx.field(static=True)
    innerOptState: optax.OptState

    # outer
    outerInfluenceTensor: JACOBIAN
    outerUoro: UORO_Param
    outerLogs: Logs
    outerTimeConstant: float = eqx.field(static=True)
    outerOptState: optax.OptState


@dataclass(frozen=True)
class GodInterpreter:
    type LocalApp[X] = App[Self, GodState, X]

    getReccurentState: LocalApp[REC_STATE]
    putReccurentState: Callable[[REC_STATE], LocalApp[Unit]]
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
    putLogs: Callable[[Logs], LocalApp[Unit]]

    getRnnParameter: LocalApp[RnnParameter]
    putRnnParameter: Callable[[RnnParameter], LocalApp[Unit]]

    getSgdParameter: LocalApp[SgdParameter]
    putSgdParameter: Callable[[SgdParameter], LocalApp[Unit]]

    getOptState: LocalApp[optax.OptState]
    putOptState: Callable[[optax.OptState], LocalApp[Unit]]
    getOptimizer: LocalApp[optax.GradientTransformation]
    getUpdater: LocalApp[Callable[[optax.Params, optax.Updates], optax.Params]]

    @do()
    def updatePRNG(self) -> G[LocalApp[PRNG]]:
        prng, new_prng = yield from gets(lambda e: prng_split(e.prng))
        _ = yield from modifies(lambda e: eqx.tree_at(lambda t: t.prng, e, new_prng))
        return pure(prng, PX[tuple[Self, GodState]]())


# def rnn_array(state: State) -> jax.Array:
#     return toVector(endowVector(state.activation))


# def to_rnn_array(state: State, rec_state: jax.Array) -> State:
#     activation = invmap(state.activation, lambda _: rec_state)
#     return eqx.tree_at(lambda t: t.activation, state, activation)


# def parameter_array(state: State) -> jax.Array:
#     return toVector(endowVector(state.rnnParameter))


# def to_parameter_array(state: State, rec_param: jax.Array) -> State:
#     rnnParameter = invmap(state.rnnParameter, lambda _: rec_param)
#     return eqx.tree_at(lambda t: t.rnnParameter, state, rnnParameter)


# def sgd_array(state: State) -> jax.Array:
#     return toVector(endowVector(state.sgdParameter))


# def to_sgd_array(state: State, sgd_param: jax.Array) -> State:
#     sgdParameter = invmap(state.sgdParameter, lambda _: sgd_param)
#     return eqx.tree_at(lambda t: t.sgdParameter, state, sgdParameter)
