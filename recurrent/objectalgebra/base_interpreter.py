from typing import Generic
from dataclasses import dataclass, replace
from recurrent.mytypes import *
from recurrent.myrecords import (
    RfloConfig,
    WithRnnConfig,
    WithBaseFuture,
    WithHyperparameter,
    WithBaseRflo,
    WithRnn,
)
from recurrent.util import rnnSplitParameters
from typing import Protocol
from recurrent.objectalgebra.typeclasses import *


@dataclass(frozen=True)
class _Rnn_Env(WithRnnConfig, WithRnn, Protocol):
    pass


_RNN_ENV = TypeVar("_RNN_ENV", bound=_Rnn_Env)


class IsActivation(
    Generic[_RNN_ENV],
    GetActivation[_RNN_ENV, ACTIVATION],
    PutActivation[_RNN_ENV, ACTIVATION],
):
    @staticmethod
    def getActivation(env: _RNN_ENV) -> ACTIVATION:
        return env.activation

    @staticmethod
    def putActivation(s: ACTIVATION, env: _RNN_ENV) -> _RNN_ENV:
        return replace(env, activation=s)


class IsParameter(
    Generic[_RNN_ENV],
    GetParameter[_RNN_ENV, PARAMETER],
    PutParameter[_RNN_ENV, PARAMETER],
):
    @staticmethod
    def getParameter(env: _RNN_ENV) -> PARAMETER:
        return env.parameter

    @staticmethod
    def putParameter(s: PARAMETER, env: _RNN_ENV) -> _RNN_ENV:
        return replace(env, parameter=s)


class IsRecurrentWeights(
    Generic[_RNN_ENV], HasRecurrentWeights[_RNN_ENV, PARAMETER, PARAMETER]
):
    @staticmethod
    def getRecurrentWeights(s: PARAMETER, env: _RNN_ENV) -> PARAMETER:
        wrec, _ = rnnSplitParameters(env, s)
        return wrec


class IsReadoutWeights(
    Generic[_RNN_ENV], HasReadoutWeights[_RNN_ENV, PARAMETER, PARAMETER]
):
    @staticmethod
    def getReadoutWeights(s: PARAMETER, env: _RNN_ENV) -> PARAMETER:
        _, wout = rnnSplitParameters(env, s)
        return wout


@dataclass(frozen=True)
class _Rnn_Learnable(
    _Rnn_Env,
    # WithTrainPrediction,
    # WithTrainLoss,
    # WithTrainGradient,
    WithHyperparameter,
    Protocol,
):
    pass


_RNN_LEARNABLE = TypeVar("_RNN_LEARNABLE", bound=_Rnn_Learnable)


class IsHyperParameter(
    Generic[_RNN_LEARNABLE],
    GetHyperParameter[_RNN_LEARNABLE, HYPERPARAMETER],
    PutHyperParameter[_RNN_LEARNABLE, HYPERPARAMETER],
):
    @staticmethod
    def getHyperParameter(env: _RNN_LEARNABLE) -> HYPERPARAMETER:
        return env.hyperparameter

    @staticmethod
    def putHyperParameter(s: HYPERPARAMETER, env: _RNN_LEARNABLE) -> _RNN_LEARNABLE:
        return replace(env, hyperparameter=s)


# class IsLoss(
#     Generic[_RNN_LEARNABLE],
#     GetLoss[_RNN_LEARNABLE, LOSS],
#     PutLoss[_RNN_LEARNABLE, LOSS],
# ):
#     @staticmethod
#     def getLoss(env: _RNN_LEARNABLE) -> LOSS:
#         return env.trainLoss

#     @staticmethod
#     def putLoss(s: LOSS, env: _RNN_LEARNABLE) -> _RNN_LEARNABLE:
#         return replace(env, trainLoss=s)


# class IsGradient(
#     Generic[_RNN_LEARNABLE],
#     GetGradient[_RNN_LEARNABLE, GRADIENT],
#     PutGradient[_RNN_LEARNABLE, GRADIENT],
# ):
#     @staticmethod
#     def getGradient(env: _RNN_LEARNABLE) -> GRADIENT:
#         return env.trainGradient

#     @staticmethod
#     def putGradient(s: GRADIENT, env: _RNN_LEARNABLE) -> _RNN_LEARNABLE:
#         return replace(env, trainGradient=s)


# class IsPrediction(
#     Generic[_RNN_LEARNABLE],
#     GetPrediction[_RNN_LEARNABLE, PREDICTION],
#     PutPrediction[_RNN_LEARNABLE, PREDICTION],
# ):
#     @staticmethod
#     def getPrediction(env: _RNN_LEARNABLE) -> PREDICTION:
#         return env.trainPrediction

#     @staticmethod
#     def putPrediction(s: PREDICTION, env: _RNN_LEARNABLE) -> _RNN_LEARNABLE:
#         return replace(env, trainPrediction=s)


@dataclass(frozen=True)
class _Rnn_Future_Cap(_Rnn_Learnable, WithBaseFuture, Protocol):
    pass


_BASE_FUTURE_CAP = TypeVar("_BASE_FUTURE_CAP", bound=_Rnn_Future_Cap)


class IsInfluenceTensor(
    Generic[_BASE_FUTURE_CAP],
    GetInfluenceTensor[_BASE_FUTURE_CAP, INFLUENCETENSOR],
    PutInfluenceTensor[_BASE_FUTURE_CAP, INFLUENCETENSOR],
):
    @staticmethod
    def getInfluenceTensor(env: _BASE_FUTURE_CAP) -> INFLUENCETENSOR:
        return env.influenceTensor

    @staticmethod
    def putInfluenceTensor(
        s: INFLUENCETENSOR, env: _BASE_FUTURE_CAP
    ) -> _BASE_FUTURE_CAP:
        return replace(env, influenceTensor=s)


@dataclass(frozen=True)
class _RfloAlgebra(_Rnn_Future_Cap, WithBaseRflo, Protocol):
    pass


_RFLO_ALGEBRA = TypeVar("_RFLO_ALGEBRA", bound=_RfloAlgebra)


class IsRflo(
    Generic[_RFLO_ALGEBRA],
    GetRfloConfig[_RFLO_ALGEBRA],
):
    @staticmethod
    def getRfloConfig(env: _RFLO_ALGEBRA) -> RfloConfig:
        return env.rfloConfig


class RnnInterpreter(
    Generic[_RNN_ENV],
    IsActivation[_RNN_ENV],
    IsParameter[_RNN_ENV],
    IsRecurrentWeights[_RNN_ENV],
    IsReadoutWeights[_RNN_ENV],
):
    pass


class RnnLearnableInterpreter(
    Generic[_RNN_LEARNABLE],
    IsActivation[_RNN_LEARNABLE],
    IsParameter[_RNN_LEARNABLE],
    IsHyperParameter[_RNN_LEARNABLE],
    IsRecurrentWeights[_RNN_LEARNABLE],
    IsReadoutWeights[_RNN_LEARNABLE],
    # IsLoss[_RNN_LEARNABLE],
    # IsGradient[_RNN_LEARNABLE],
    # IsPrediction[_RNN_LEARNABLE],
):
    pass


# todo: messed up naming, should be past facing
class RnnWithFutureInterpreter(
    Generic[_BASE_FUTURE_CAP],
    IsActivation[_BASE_FUTURE_CAP],
    IsParameter[_BASE_FUTURE_CAP],
    IsHyperParameter[_BASE_FUTURE_CAP],
    IsRecurrentWeights[_BASE_FUTURE_CAP],
    IsReadoutWeights[_BASE_FUTURE_CAP],
    IsInfluenceTensor[_BASE_FUTURE_CAP],
    # IsLoss[_BASE_FUTURE_CAP],
    # IsGradient[_BASE_FUTURE_CAP],
    # IsPrediction[_BASE_FUTURE_CAP],
):
    pass


class RfloInterpreter(
    Generic[_RFLO_ALGEBRA],
    IsRflo[_RFLO_ALGEBRA],
    RnnWithFutureInterpreter[_RFLO_ALGEBRA],
):
    pass
