from dataclasses import dataclass
from typing import Self
from donotation import do

from recurrent.datarecords import *
from recurrent.monad import *
from recurrent.mytypes import *
from recurrent.parameters import *
from recurrent.util import prng_split


@dataclass(frozen=True)
class State(eqx.Module):
    activation: ACTIVATION
    influenceTensor: JACOBIAN
    rnnParameter: RnnParameter
    sgdParameter: SgdParameter
    uoro: UORO_Param
    logs: Logs
    rnnConfig: RnnConfig = eqx.field(static=True)


@dataclass(frozen=True)
class GodState(eqx.Module):
    states: list[State]
    prng: PRNG
    logConfig: LogConfig = eqx.field(static=True)


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
    getLogConfig: LocalApp[LogConfig]
    putLogs: Callable[[Logs], LocalApp[Unit]]

    getRnnParameter: LocalApp[RnnParameter]
    putRnnParameter: Callable[[RnnParameter], LocalApp[Unit]]
    getSgdParameter: LocalApp[SgdParameter]
    putSgdParameter: Callable[[SgdParameter], LocalApp[Unit]]

    @do()
    def updatePRNG(self) -> G[LocalApp[PRNG]]:
        prng, new_prng = yield from gets(lambda e: prng_split(e.prng))
        _ = yield from modifies(lambda e: eqx.tree_at(lambda t: t.prng, e, new_prng))
        return pure(prng, PX[tuple[Self, GodState]]())
