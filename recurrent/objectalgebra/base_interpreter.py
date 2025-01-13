from typing import Generic
from dataclasses import dataclass, replace
from recurrent.mytypes import *
from recurrent.mixins import (
    RnnParameter,
    SgdParameter,
    WithBasePast,
    WithRnnActivation,
    WithBaseRflo,
    WithRnnParameter,
    WithSgdParameter,
)
from recurrent.parameters import RfloConfig
from typing import Protocol
from recurrent.objectalgebra.typeclasses import *


_HAS_ACTIVATION = TypeVar("_HAS_ACTIVATION", bound=WithRnnActivation)


class BaseActivation(
    Generic[_HAS_ACTIVATION],
    GetActivation[_HAS_ACTIVATION, ACTIVATION],
    PutActivation[_HAS_ACTIVATION, ACTIVATION],
):
    def getActivation(self, env: _HAS_ACTIVATION) -> ACTIVATION:
        return env.activation

    def putActivation(self, z: ACTIVATION, env: _HAS_ACTIVATION) -> _HAS_ACTIVATION:
        return replace(env, activation=z)


_IS_RNNP = TypeVar("_IS_RNNP", bound=WithRnnParameter)


class BaseRnnParameterRead(Generic[_IS_RNNP], GetParameter[_IS_RNNP, RnnParameter]):
    def getParameter(self, env: _IS_RNNP) -> RnnParameter:
        return env.parameter


class BaseRnnParameterWrite(Generic[_IS_RNNP], PutParameter[_IS_RNNP, RnnParameter]):
    def putParameter(self, z: RnnParameter, env: _IS_RNNP) -> _IS_RNNP:
        return replace(env, parameter=z)


_IS_SGDP = TypeVar("_IS_SGDP", bound=WithSgdParameter)


class BaseHyperparameter(
    Generic[_IS_SGDP],
    GetHyperParameter[_IS_SGDP, SgdParameter],
    PutHyperParameter[_IS_SGDP, SgdParameter],
):
    def getHyperParameter(self, env: _IS_SGDP) -> SgdParameter:
        return env.hyperparameter

    def putHyperParameter(self, z: SgdParameter, env: _IS_SGDP) -> _IS_SGDP:
        return replace(env, hyperparameter=z)


_HAS_INFLUENCE = TypeVar("_HAS_INFLUENCE", bound=WithBasePast)


class BaseInfluenceTensor(
    Generic[_HAS_INFLUENCE],
    GetInfluenceTensor[_HAS_INFLUENCE, INFLUENCETENSOR],
    PutInfluenceTensor[_HAS_INFLUENCE, INFLUENCETENSOR],
):
    def getInfluenceTensor(self, env: _HAS_INFLUENCE) -> INFLUENCETENSOR:
        return env.influenceTensor

    def putInfluenceTensor(
        self, z: INFLUENCETENSOR, env: _HAS_INFLUENCE
    ) -> _HAS_INFLUENCE:
        return replace(env, influenceTensor=z)


_RFLO_ALGEBRA = TypeVar("_RFLO_ALGEBRA", bound=WithBaseRflo)


class BaseRflo(
    Generic[_RFLO_ALGEBRA],
    GetRfloConfig[_RFLO_ALGEBRA],
):
    def getRfloConfig(self, env: _RFLO_ALGEBRA) -> RfloConfig:
        return env.rfloConfig


@dataclass(frozen=True, slots=True)
class _Inference(WithRnnActivation, WithRnnParameter, Protocol):
    pass


_INFERENCE = TypeVar("_INFERENCE", bound=_Inference)


class RnnInferenceInterpreter(
    Generic[_INFERENCE], BaseActivation[_INFERENCE], BaseRnnParameterRead[_INFERENCE]
):
    pass


@dataclass(frozen=True, slots=True)
class _Learnable(WithRnnActivation, WithRnnParameter, WithSgdParameter, Protocol):
    pass


_LEARNABLE = TypeVar("_LEARNABLE", bound=_Learnable)


class RnnLearnableInterpreter(
    Generic[_LEARNABLE],
    BaseActivation[_LEARNABLE],
    BaseRnnParameterRead[_LEARNABLE],
    BaseRnnParameterWrite[_LEARNABLE],
    BaseHyperparameter[_LEARNABLE],
):
    pass


@dataclass(frozen=True, slots=True)
class _PastFacing(_Learnable, WithBasePast, Protocol):
    pass


_PAST_FACING = TypeVar("_PAST_FACING", bound=_PastFacing)


class RnnPastFacingInterpreter(
    Generic[_PAST_FACING],
    BaseActivation[_PAST_FACING],
    BaseRnnParameterRead[_PAST_FACING],
    BaseRnnParameterWrite[_PAST_FACING],
    BaseHyperparameter[_PAST_FACING],
    BaseInfluenceTensor[_PAST_FACING],
):
    pass


@dataclass(frozen=True, slots=True)
class _Rflo(_PastFacing, WithBaseRflo, Protocol):
    pass


_RFLO = TypeVar("_RFLO", bound=_Rflo)


class RfloInterpreter(
    Generic[_RFLO],
    BaseRflo[_RFLO],
    RnnPastFacingInterpreter[_RFLO],
):
    pass
