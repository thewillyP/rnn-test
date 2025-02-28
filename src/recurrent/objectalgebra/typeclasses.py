from typing import Self
from recurrent.myrecords import GodState
from recurrent.mytypes import *
from recurrent.parameters import *
from typing import Protocol
from recurrent.monad import *


# ============== Typeclasses ==============


class GetRecurrentState(Protocol):
    @property
    def getRecurrentState(self) -> App[Self, GodState, REC_STATE]: ...


class PutRecurrentState(Protocol):
    def putRecurrentState(self, s: REC_STATE) -> App[Self, GodState, Unit]: ...


class GetRecurrentParam(Protocol):
    @property
    def getRecurrentParam(self) -> App[Self, GodState, REC_PARAM]: ...


class PutRecurrentParam(Protocol):
    def putRecurrentParam(self, s: REC_PARAM) -> App[Self, GodState, Unit]: ...


class GetActivation(Protocol):
    @property
    def getActivation(self) -> App[Self, GodState, ACTIVATION]: ...


class PutActivation(Protocol):
    def putActivation(self, s: ACTIVATION) -> App[Self, GodState, Unit]: ...


class GetRnnParameter(Protocol):
    @property
    def getRnnParameter(self) -> App[Self, GodState, RnnParameter]: ...


class PutRnnParameter(Protocol):
    def putRnnParameter(self, s: RnnParameter) -> App[Self, GodState, Unit]: ...


class GetSgdParameter(Protocol):
    @property
    def getSgdParameter(self) -> App[Self, GodState, SgdParameter]: ...


class PutSgdParameter(Protocol):
    def putSgdParameter(self, s: SgdParameter) -> App[Self, GodState, Unit]: ...


class GetInfluenceTensor(Protocol):
    @property
    def getInfluenceTensor(self) -> App[Self, GodState, JACOBIAN]: ...


class PutInfluenceTensor(Protocol):
    def putInfluenceTensor(self, s: JACOBIAN) -> App[Self, GodState, Unit]: ...


class GetUoro(Protocol):
    @property
    def getUoro(self) -> App[Self, GodState, UORO_Param]: ...


class PutUoro(Protocol):
    def putUoro(self, s: UORO_Param) -> App[Self, GodState, Unit]: ...


class GetRnnConfig(Protocol):
    @property
    def getRnnConfig(self) -> App[Self, GodState, RnnConfig]: ...


class PutLogs(Protocol):
    def putLogs(self, s: Logs) -> App[Self, GodState, Unit]: ...


class GetLogConfig(Protocol):
    @property
    def getLogConfig(self) -> App[Self, GodState, LogConfig]: ...


class GetPRNG(Protocol):
    def updatePRNG(self) -> App[Self, GodState, PRNG]: ...


class GetTimeConstant(Protocol):
    @property
    def getTimeConstant(self) -> App[Self, GodState, float]: ...
