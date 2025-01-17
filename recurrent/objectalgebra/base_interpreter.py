from typing import Generic, NamedTuple, Self
from recurrent.myrecords import RnnGodState
from recurrent.mytypes import *
from recurrent.parameters import RfloConfig
from typing import Protocol
from recurrent.objectalgebra.typeclasses import *
import copy


class BaseRnnGodInterpreter(
    Generic[A, B, C],
    GetActivation[RnnGodState[A, B, C], ACTIVATION],
    PutActivation[RnnGodState[A, B, C], ACTIVATION],
    GetParameter[RnnGodState[A, B, C], A],
    PutParameter[RnnGodState[A, B, C], A],
    GetHyperParameter[RnnGodState[A, B, C], B],
    PutHyperParameter[RnnGodState[A, B, C], B],
    GetInfluenceTensor[RnnGodState[A, B, C], INFLUENCETENSOR],
    PutInfluenceTensor[RnnGodState[A, B, C], INFLUENCETENSOR],
    GetRfloConfig[RnnGodState[A, B, C]],
):
    def getActivation(self, env: RnnGodState) -> ACTIVATION:
        return env.activation

    def putActivation(self, z: ACTIVATION, env: RnnGodState) -> RnnGodState:
        return copy.replace(env, activation=z)

    def getParameter(self, env: RnnGodState[A, B, C]) -> A:
        return env.parameter

    def putParameter(self, z: A, env: RnnGodState[A, B, C]) -> RnnGodState[A, B, C]:
        return copy.replace(env, parameter=z)

    def getHyperParameter(self, env: RnnGodState[A, B, C]) -> B:
        return env.hyperparameter

    def putHyperParameter(
        self, z: B, env: RnnGodState[A, B, C]
    ) -> RnnGodState[A, B, C]:
        return copy.replace(env, hyperparameter=z)

    def getInfluenceTensor(self, env: RnnGodState[A, B, C]) -> INFLUENCETENSOR:
        return env.influenceTensor

    def putInfluenceTensor(
        self, z: INFLUENCETENSOR, env: RnnGodState[A, B, C]
    ) -> RnnGodState[A, B, C]:
        return copy.replace(env, influenceTensor=z)

    def getRfloConfig(self, env: RnnGodState[A, B, C]) -> RfloConfig:
        return env.rfloConfig


class _Dialect(
    Generic[ENV, T, E],
    GetParameter[ENV, T],
    PutParameter[ENV, T],
    GetHyperParameter[ENV, E],
    PutHyperParameter[ENV, E],
):
    pass


class BilevelRnnGodInterpreter(
    Generic[A, B, C],
    GetActivation[RnnGodState[A, B, C], A],
    PutActivation[RnnGodState[A, B, C], A],
    GetParameter[RnnGodState[A, B, C], B],
    PutParameter[RnnGodState[A, B, C], B],
    GetHyperParameter[RnnGodState[A, B, C], C],
    PutHyperParameter[RnnGodState[A, B, C], C],
    GetInfluenceTensor[RnnGodState[A, B, C], INFLUENCETENSOR],
    PutInfluenceTensor[RnnGodState[A, B, C], INFLUENCETENSOR],
    GetRfloConfig[RnnGodState[A, B, C]],
):
    def __init__(self, dialect: _Dialect[RnnGodState[A, B, C], A, B]) -> None:
        super().__init__()
        self.dialect = dialect

    def getActivation(self, env: RnnGodState) -> A:
        return self.dialect.getParameter(env)

    def putActivation(self, z: A, env: RnnGodState) -> RnnGodState:
        return self.dialect.putParameter(z, env)

    def getParameter(self, env: RnnGodState[A, B, C]) -> B:
        return self.dialect.getHyperParameter(env)

    def putParameter(self, z: B, env: RnnGodState[A, B, C]) -> RnnGodState[A, B, C]:
        return self.dialect.putHyperParameter(z, env)

    def getHyperParameter(self, env: RnnGodState[A, B, C]) -> C:
        return env.metaHyperparameter

    def putHyperParameter(
        self, z: C, env: RnnGodState[A, B, C]
    ) -> RnnGodState[A, B, C]:
        return copy.replace(env, metaHyperparameter=z)

    def getInfluenceTensor(self, env: RnnGodState[A, B, C]) -> INFLUENCETENSOR:
        return env.ohoInfluenceTensor

    def putInfluenceTensor(
        self, z: INFLUENCETENSOR, env: RnnGodState[A, B, C]
    ) -> RnnGodState[A, B, C]:
        return copy.replace(env, ohoInfluenceTensor=z)

    def getRfloConfig(self, env: RnnGodState[A, B, C]) -> RfloConfig:
        return env.rfloConfig_bilevel
