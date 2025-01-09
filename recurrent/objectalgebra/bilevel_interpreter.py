from typing import Generic
from dataclasses import dataclass, replace
from recurrent.mytypes import *
from recurrent.myrecords import (
    WithBilevel,
    WithOhoFuture,
    WithValidationGradient,
    WithValidationLoss,
    WithValidationPrediction,
    WithBilevelRflo,
)
from typing import Protocol
from recurrent.objectalgebra.base_interpreter import (
    _Rnn_Learnable,
    RnnLearnableInterpreter,
)
from recurrent.objectalgebra.typeclasses import *


@dataclass(frozen=True)
class _Bilevel_Learnable(
    _Rnn_Learnable,
    WithBilevel,
    WithValidationPrediction,
    WithValidationLoss,
    WithValidationGradient,
    Protocol,
):
    pass


_BILEVEL_LEARNABLE = TypeVar("_BILEVEL_LEARNABLE", bound=_Bilevel_Learnable)


class IsActivationOHO(
    Generic[_BILEVEL_LEARNABLE],
    GetActivation[_BILEVEL_LEARNABLE, PARAMETER],
    PutActivation[_BILEVEL_LEARNABLE, PARAMETER],
):
    getActivation = staticmethod(
        RnnLearnableInterpreter[_BILEVEL_LEARNABLE].getParameter
    )
    putActivation = staticmethod(
        RnnLearnableInterpreter[_BILEVEL_LEARNABLE].putParameter
    )


class IsParameterOHO(
    Generic[_BILEVEL_LEARNABLE],
    GetParameter[_BILEVEL_LEARNABLE, HYPERPARAMETER],
    PutParameter[_BILEVEL_LEARNABLE, HYPERPARAMETER],
):
    getParameter = staticmethod(
        RnnLearnableInterpreter[_BILEVEL_LEARNABLE].getHyperParameter
    )
    putParameter = staticmethod(
        RnnLearnableInterpreter[_BILEVEL_LEARNABLE].putHyperParameter
    )


# class IsLossOHO(
#     Generic[_BILEVEL_LEARNABLE],
#     GetLoss[_BILEVEL_LEARNABLE, LOSS],
#     PutLoss[_BILEVEL_LEARNABLE, LOSS],
# ):
#     @staticmethod
#     def getLoss(env: _BILEVEL_LEARNABLE) -> LOSS:
#         return env.validationLoss

#     @staticmethod
#     def putLoss(s: LOSS, env: _BILEVEL_LEARNABLE) -> _BILEVEL_LEARNABLE:
#         return replace(env, validationLoss=s)


# class IsGradientOHO(
#     Generic[_BILEVEL_LEARNABLE],
#     GetGradient[_BILEVEL_LEARNABLE, GRADIENT],
#     PutGradient[_BILEVEL_LEARNABLE, GRADIENT],
# ):
#     @staticmethod
#     def getGradient(env: _BILEVEL_LEARNABLE) -> GRADIENT:
#         return env.validationGradient

#     @staticmethod
#     def putGradient(s: GRADIENT, env: _BILEVEL_LEARNABLE) -> _BILEVEL_LEARNABLE:
#         return replace(env, validationGradient=s)


# class IsPredictionOHO(
#     Generic[_BILEVEL_LEARNABLE],
#     GetPrediction[_BILEVEL_LEARNABLE, PREDICTION],
#     PutPrediction[_BILEVEL_LEARNABLE, PREDICTION],
# ):
#     @staticmethod
#     def getPrediction(env: _BILEVEL_LEARNABLE) -> PREDICTION:
#         return env.validationPrediction

#     @staticmethod
#     def putPrediction(s: PREDICTION, env: _BILEVEL_LEARNABLE) -> _BILEVEL_LEARNABLE:
#         return replace(env, validationPrediction=s)


class IsHyperParameterOHO(
    Generic[_BILEVEL_LEARNABLE],
    GetHyperParameter[_BILEVEL_LEARNABLE, METAHYPERPARAMETER],
    PutHyperParameter[_BILEVEL_LEARNABLE, METAHYPERPARAMETER],
):
    @staticmethod
    def getHyperParameter(env: _BILEVEL_LEARNABLE) -> METAHYPERPARAMETER:
        return env.metaHyperparameter

    @staticmethod
    def putHyperParameter(
        s: METAHYPERPARAMETER, env: _BILEVEL_LEARNABLE
    ) -> _BILEVEL_LEARNABLE:
        return replace(env, metaHyperparameter=s)


class IsRecurrentWeightsOHO(
    Generic[_BILEVEL_LEARNABLE],
    HasRecurrentWeights[_BILEVEL_LEARNABLE, HYPERPARAMETER, HYPERPARAMETER],
):
    @staticmethod
    def getRecurrentWeights(
        hyperparameter: HYPERPARAMETER, _: _BILEVEL_LEARNABLE
    ) -> HYPERPARAMETER:
        return hyperparameter


class IsReadoutWeightsOHO(
    Generic[_BILEVEL_LEARNABLE],
    HasReadoutWeights[_BILEVEL_LEARNABLE, HYPERPARAMETER, PARAMETER],
):
    @staticmethod
    def getReadoutWeights(_: HYPERPARAMETER, env: _BILEVEL_LEARNABLE) -> PARAMETER:
        return RnnLearnableInterpreter[_BILEVEL_LEARNABLE].getParameter(env)


@dataclass(frozen=True)
class _Oho_Future(_Bilevel_Learnable, WithOhoFuture, Protocol):
    pass


_OHO_FUTURE = TypeVar("_OHO_FUTURE", bound=_Oho_Future)


class IsInfluenceTensorOHO(
    Generic[_OHO_FUTURE],
    GetInfluenceTensor[_OHO_FUTURE, INFLUENCETENSOR],
    PutInfluenceTensor[_OHO_FUTURE, INFLUENCETENSOR],
):
    @staticmethod
    def getInfluenceTensor(env: _OHO_FUTURE) -> INFLUENCETENSOR:
        return env.ohoInfluenceTensor

    @staticmethod
    def putInfluenceTensor(s: INFLUENCETENSOR, env: _OHO_FUTURE) -> _OHO_FUTURE:
        return replace(env, ohoInfluenceTensor=s)


@dataclass(frozen=True)
class _RfloAlgebra(_Oho_Future, WithBilevelRflo, Protocol):
    pass


_RFLO_ALGEBRA = TypeVar("_RFLO_ALGEBRA", bound=_RfloAlgebra)


class IsRflo(
    Generic[_RFLO_ALGEBRA],
    GetRfloConfig[_RFLO_ALGEBRA],
):
    @staticmethod
    def getRfloConfig(env: _RFLO_ALGEBRA) -> RfloConfig:
        return env.rfloConfig_bilevel


class BilevelInterpreter(
    Generic[_BILEVEL_LEARNABLE],
    IsActivationOHO[_BILEVEL_LEARNABLE],
    IsParameterOHO[_BILEVEL_LEARNABLE],
    IsHyperParameterOHO[_BILEVEL_LEARNABLE],
    IsRecurrentWeightsOHO[_BILEVEL_LEARNABLE],
    IsReadoutWeightsOHO[_BILEVEL_LEARNABLE],
    # IsLossOHO[_BILEVEL_LEARNABLE],
    # IsGradientOHO[_BILEVEL_LEARNABLE],
    # IsPredictionOHO[_BILEVEL_LEARNABLE],
):
    pass


class BilevelWithOhoInterpreter(
    Generic[_OHO_FUTURE],
    IsActivationOHO[_OHO_FUTURE],
    IsParameterOHO[_OHO_FUTURE],
    IsHyperParameterOHO[_OHO_FUTURE],
    IsRecurrentWeightsOHO[_OHO_FUTURE],
    IsReadoutWeightsOHO[_OHO_FUTURE],
    IsInfluenceTensorOHO[_OHO_FUTURE],
    # IsLossOHO[_OHO_FUTURE],
    # IsGradientOHO[_OHO_FUTURE],
    # IsPredictionOHO[_OHO_FUTURE],
):
    pass


class RfloInterpreter(
    Generic[_RFLO_ALGEBRA],
    IsRflo[_RFLO_ALGEBRA],
    BilevelWithOhoInterpreter[_RFLO_ALGEBRA],
):
    pass
