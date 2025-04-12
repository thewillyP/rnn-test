from typing import Self
import optax
from recurrent.mytypes import *
from recurrent.parameters import *
from typing import Protocol
from recurrent.monad import *


# ============== Typeclasses ==============


class GetRecurrentState[Env](Protocol):
    @property
    def getRecurrentState(self) -> App[Self, Env, REC_STATE]: ...


class PutRecurrentState[Env](Protocol):
    def putRecurrentState(self, s: REC_STATE) -> App[Self, Env, Unit]: ...


class GetRecurrentParam[Env](Protocol):
    @property
    def getRecurrentParam(self) -> App[Self, Env, REC_PARAM]: ...


class PutRecurrentParam[Env](Protocol):
    def putRecurrentParam(self, s: REC_PARAM) -> App[Self, Env, Unit]: ...


class GetActivation[Env](Protocol):
    @property
    def getActivation(self) -> App[Self, Env, ACTIVATION]: ...


class PutActivation[Env](Protocol):
    def putActivation(self, s: ACTIVATION) -> App[Self, Env, Unit]: ...


class GetRnnParameter[Env](Protocol):
    @property
    def getRnnParameter(self) -> App[Self, Env, RnnParameter]: ...


class PutRnnParameter[Env](Protocol):
    def putRnnParameter(self, s: RnnParameter) -> App[Self, Env, Unit]: ...


class GetSgdParameter[Env](Protocol):
    @property
    def getSgdParameter(self) -> App[Self, Env, SgdParameter]: ...


class PutSgdParameter[Env](Protocol):
    def putSgdParameter(self, s: SgdParameter) -> App[Self, Env, Unit]: ...


class GetOptimizer[Env](Protocol):
    @property
    def getOptimizer(self) -> App[Self, Env, optax.GradientTransformation]: ...


class GetOptState[Env](Protocol):
    @property
    def getOptState(self) -> App[Self, Env, optax.OptState]: ...


class PutOptState[Env](Protocol):
    def putOptState(self, s: optax.OptState) -> App[Self, Env, Unit]: ...


class GetUpdater[Env](Protocol):
    @property
    def getUpdater(self) -> App[Self, Env, Callable[[optax.Params, optax.Updates], optax.Params]]: ...


class GetInfluenceTensor[Env](Protocol):
    @property
    def getInfluenceTensor(self) -> App[Self, Env, JACOBIAN]: ...


class PutInfluenceTensor[Env](Protocol):
    def putInfluenceTensor(self, s: JACOBIAN) -> App[Self, Env, Unit]: ...


class GetUoro[Env](Protocol):
    @property
    def getUoro(self) -> App[Self, Env, UORO_Param]: ...


class PutUoro[Env](Protocol):
    def putUoro(self, s: UORO_Param) -> App[Self, Env, Unit]: ...


class GetRnnConfig[Env](Protocol):
    @property
    def getRnnConfig(self) -> App[Self, Env, RnnConfig]: ...


class PutLogs[Env](Protocol):
    def putLogs(self, s: Logs) -> App[Self, Env, Unit]: ...


class GetLogConfig[Env](Protocol):
    @property
    def getLogConfig(self) -> App[Self, Env, LogConfig]: ...


class GetGlobalLogConfig[Env](Protocol):
    @property
    def getGlobalLogConfig(self) -> App[Self, Env, GlobalLogConfig]: ...


class GetPRNG[Env](Protocol):
    def updatePRNG(self) -> App[Self, Env, PRNG]: ...


class GetTimeConstant[Env](Protocol):
    @property
    def getTimeConstant(self) -> App[Self, Env, float]: ...
