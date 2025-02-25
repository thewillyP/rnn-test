from typing import Generic, Self
from recurrent.datarecords import DataGod, InputOutput, OhoInputOutput
from recurrent.myrecords import GodState
from recurrent.mytypes import *
from recurrent.parameters import *
from typing import Protocol
from recurrent.monad import *


# ============== Typeclasses ==============


class GetRecurrentState(Protocol):
    @property
    def getRecurrentState(self) -> App[Self, DataGod, GodState, REC_STATE]: ...


class PutRecurrentState(Protocol):
    def putRecurrentState(self, s: REC_STATE) -> App[Self, DataGod, GodState, Unit]: ...


class GetRecurrentParam(Protocol):
    @property
    def getRecurrentParam(self) -> App[Self, DataGod, GodState, REC_PARAM]: ...


class PutRecurrentParam(Protocol):
    def putRecurrentParam(self, s: REC_PARAM) -> App[Self, DataGod, GodState, Unit]: ...


class GetActivation(Protocol):
    @property
    def getActivation(self) -> App[Self, DataGod, GodState, ACTIVATION]: ...


class PutActivation(Protocol):
    def putActivation(self, s: ACTIVATION) -> App[Self, DataGod, GodState, Unit]: ...


class GetRnnParameter(Protocol):
    @property
    def getRnnParameter(self) -> App[Self, DataGod, GodState, RnnParameter]: ...


class PutRnnParameter(Protocol):
    def putRnnParameter(self, s: RnnParameter) -> App[Self, DataGod, GodState, Unit]: ...


class GetSgdParameter(Protocol):
    @property
    def getSgdParameter(self) -> App[Self, DataGod, GodState, SgdParameter]: ...


class PutSgdParameter(Protocol):
    def putSgdParameter(self, s: SgdParameter) -> App[Self, DataGod, GodState, Unit]: ...


class GetInfluenceTensor(Protocol):
    @property
    def getInfluenceTensor(self) -> App[Self, DataGod, GodState, JACOBIAN]: ...


class PutInfluenceTensor(Protocol):
    def putInfluenceTensor(self, s: JACOBIAN) -> App[Self, DataGod, GodState, Unit]: ...


class GetUoro(Protocol):
    @property
    def getUoro(self) -> App[Self, DataGod, GodState, UORO_Param]: ...


class PutUoro(Protocol):
    def putUoro(self, s: UORO_Param) -> App[Self, DataGod, GodState, Unit]: ...


class GetRnnConfig(Protocol):
    @property
    def getRnnConfig(self) -> App[Self, DataGod, GodState, RnnConfig]: ...


class PutLogs(Protocol):
    def putLogs(self, s: Logs) -> App[Self, DataGod, GodState, Unit]: ...


class GetLogConfig(Protocol):
    @property
    def getLogConfig(self) -> App[Self, DataGod, GodState, LogConfig]: ...


class GetInput(Protocol):
    @property
    def getInput(self) -> App[Self, DataGod, GodState, INPUT]: ...


class GetLabel(Protocol):
    @property
    def getLabel(self) -> App[Self, DataGod, GodState, LABEL]: ...


class GetPredictionInput(Protocol):
    @property
    def getPredictionInput(self) -> App[Self, DataGod, GodState, PREDICTION_INPUT]: ...


class GetInputOutput(Protocol):
    @property
    def getInputOutput(self) -> App[Self, DataGod, GodState, InputOutput]: ...


class GetOhoInputOutput(Protocol):
    @property
    def getOhoInputOutput(self) -> App[Self, DataGod, GodState, OhoInputOutput]: ...


class GetPRNG(Protocol):
    def updatePRNG(self) -> App[Self, DataGod, GodState, PRNG]: ...
