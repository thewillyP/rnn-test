from typing import Generic
from dataclasses import dataclass, replace
from mytypes import *
from myrecords import (
    RnnConfig,
    WithBaseFuture,
    WithBilevel,
    WithHyperparameter,
    WithOhoFuture,
    WithRnn,
    WithTrainGradient,
    WithTrainLoss,
    WithTrainPrediction,
    WithValidationGradient,
    WithValidationLoss,
    WithValidationPrediction,
)
from util import rnnSplitParameters
from typing import Protocol

_ENV = TypeVar("_ENV", contravariant=True)
_DATA = TypeVar("_DATA", contravariant=True)
_T = TypeVar("_T", contravariant=True)
_E = TypeVar("_E", covariant=True)


# ============== Typeclasses ==============
class GetActivation(Generic[_ENV, _E], Protocol):
    @staticmethod
    def getActivation(env: _ENV) -> _E:
        pass


class PutActivation(Generic[ENV, _T], Protocol):
    @staticmethod
    def putActivation(s: _T, env: ENV) -> ENV:
        pass


class GetParameter(Generic[_ENV, _E], Protocol):
    @staticmethod
    def getParameter(env: _ENV) -> _E:
        pass


class PutParameter(Generic[ENV, _T], Protocol):
    @staticmethod
    def putParameter(s: _T, env: ENV) -> ENV:
        pass


class GetHyperParameter(Generic[_ENV, _E], Protocol):
    @staticmethod
    def getHyperParameter(env: _ENV) -> _E:
        pass


class PutHyperParameter(Generic[ENV, _T], Protocol):
    @staticmethod
    def putHyperParameter(s: _T, env: ENV) -> ENV:
        pass


class GetInfluenceTensor(Generic[_ENV, _E], Protocol):
    @staticmethod
    def getInfluenceTensor(env: _ENV) -> _E:
        pass


class PutInfluenceTensor(Generic[ENV, _T], Protocol):
    @staticmethod
    def putInfluenceTensor(s: _T, env: ENV) -> ENV:
        pass


class HasRecurrentWeights(Protocol[_ENV, _T, _E]):
    @staticmethod
    def getRecurrentWeights(s: _T, env: _ENV) -> _E:
        pass


class HasReadoutWeights(Generic[_ENV, _T, _E], Protocol):
    @staticmethod
    def getReadoutWeights(s: _T, env: _ENV) -> _E:
        pass


class GetLoss(Generic[_ENV, _E], Protocol):
    @staticmethod
    def getLoss(env: _ENV) -> _E:
        pass


class PutLoss(Generic[ENV, _T], Protocol):
    @staticmethod
    def putLoss(s: _T, env: ENV) -> ENV:
        pass


class GetGradient(Generic[_ENV, _E], Protocol):
    @staticmethod
    def getGradient(env: _ENV) -> _E:
        pass


class PutGradient(Generic[ENV, _T], Protocol):
    @staticmethod
    def putGradient(s: _T, env: ENV) -> ENV:
        pass


class GetPrediction(Generic[_ENV, _E], Protocol):
    @staticmethod
    def getPrediction(env: _ENV) -> _E:
        pass


class PutPrediction(Generic[ENV, _T], Protocol):
    @staticmethod
    def putPrediction(s: _T, env: ENV) -> ENV:
        pass


class HasInput(Generic[_DATA, _E], Protocol):
    @staticmethod
    def getInput(data: _DATA) -> _E:
        pass


class HasPredictionInput(Generic[_DATA, _E], Protocol):
    @staticmethod
    def getPredictionInput(data: _DATA) -> _E:
        pass


class HasLabel(Generic[_DATA, _E], Protocol):
    @staticmethod
    def getLabel(data: _DATA) -> _E:
        pass


# class HasValidationInput(Generic[_DATA, T], metaclass=ABCMeta):
#     @abstractmethod
#     @staticmethod
#     def getValidationInput(data: _DATA) -> T:
#         pass

# class HasValidationLabel(Generic[_DATA, T], metaclass=ABCMeta):
#     @abstractmethod
#     @staticmethod
#     def getValidationLabel(data: _DATA) -> T:
#         pass


# ============== Instance ==============

# downside is that I need to create a whole new class that inherits all the instantiations.
# I will just maintain and reedit my god instance interpreter. Best compromise. Not truly extensible but I have full control.

# @dataclass(frozen=True)
# class OhoState(Generic[A, B, C]):
#     activation: A
#     parameter: B
#     hyperparameter: C


@dataclass(frozen=True)
class _Rnn_Env(RnnConfig, WithRnn, Protocol):
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
    WithTrainPrediction,
    WithTrainLoss,
    WithTrainGradient,
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


class IsLoss(
    Generic[_RNN_LEARNABLE],
    GetLoss[_RNN_LEARNABLE, LOSS],
    PutLoss[_RNN_LEARNABLE, LOSS],
):
    @staticmethod
    def getLoss(env: _RNN_LEARNABLE) -> LOSS:
        return env.trainLoss

    @staticmethod
    def putLoss(s: LOSS, env: _RNN_LEARNABLE) -> _RNN_LEARNABLE:
        return replace(env, trainLoss=s)


class IsGradient(
    Generic[_RNN_LEARNABLE],
    GetGradient[_RNN_LEARNABLE, GRADIENT],
    PutGradient[_RNN_LEARNABLE, GRADIENT],
):
    @staticmethod
    def getGradient(env: _RNN_LEARNABLE) -> GRADIENT:
        return env.trainGradient

    @staticmethod
    def putGradient(s: GRADIENT, env: _RNN_LEARNABLE) -> _RNN_LEARNABLE:
        return replace(env, trainGradient=s)


class IsPrediction(
    Generic[_RNN_LEARNABLE],
    GetPrediction[_RNN_LEARNABLE, PREDICTION],
    PutPrediction[_RNN_LEARNABLE, PREDICTION],
):
    @staticmethod
    def getPrediction(env: _RNN_LEARNABLE) -> PREDICTION:
        return env.trainPrediction

    @staticmethod
    def putPrediction(s: PREDICTION, env: _RNN_LEARNABLE) -> _RNN_LEARNABLE:
        return replace(env, trainPrediction=s)


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
    IsLoss[_RNN_LEARNABLE],
    IsGradient[_RNN_LEARNABLE],
    IsPrediction[_RNN_LEARNABLE],
):
    pass


class RnnWithFutureInterpreter(
    Generic[_BASE_FUTURE_CAP],
    IsActivation[_BASE_FUTURE_CAP],
    IsParameter[_BASE_FUTURE_CAP],
    IsHyperParameter[_BASE_FUTURE_CAP],
    IsRecurrentWeights[_BASE_FUTURE_CAP],
    IsReadoutWeights[_BASE_FUTURE_CAP],
    IsInfluenceTensor[_BASE_FUTURE_CAP],
    IsLoss[_BASE_FUTURE_CAP],
    IsGradient[_BASE_FUTURE_CAP],
    IsPrediction[_BASE_FUTURE_CAP],
):
    pass


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


class IsLossOHO(
    Generic[_BILEVEL_LEARNABLE],
    GetLoss[_BILEVEL_LEARNABLE, LOSS],
    PutLoss[_BILEVEL_LEARNABLE, LOSS],
):
    @staticmethod
    def getLoss(env: _BILEVEL_LEARNABLE) -> LOSS:
        return env.validationLoss

    @staticmethod
    def putLoss(s: LOSS, env: _BILEVEL_LEARNABLE) -> _BILEVEL_LEARNABLE:
        return replace(env, validationLoss=s)


class IsGradientOHO(
    Generic[_BILEVEL_LEARNABLE],
    GetGradient[_BILEVEL_LEARNABLE, GRADIENT],
    PutGradient[_BILEVEL_LEARNABLE, GRADIENT],
):
    @staticmethod
    def getGradient(env: _BILEVEL_LEARNABLE) -> GRADIENT:
        return env.validationGradient

    @staticmethod
    def putGradient(s: GRADIENT, env: _BILEVEL_LEARNABLE) -> _BILEVEL_LEARNABLE:
        return replace(env, validationGradient=s)


class IsPredictionOHO(
    Generic[_BILEVEL_LEARNABLE],
    GetPrediction[_BILEVEL_LEARNABLE, PREDICTION],
    PutPrediction[_BILEVEL_LEARNABLE, PREDICTION],
):
    @staticmethod
    def getPrediction(env: _BILEVEL_LEARNABLE) -> PREDICTION:
        return env.validationPrediction

    @staticmethod
    def putPrediction(s: PREDICTION, env: _BILEVEL_LEARNABLE) -> _BILEVEL_LEARNABLE:
        return replace(env, validationPrediction=s)


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


class BilevelInterpreter(
    Generic[_BILEVEL_LEARNABLE],
    IsActivationOHO[_BILEVEL_LEARNABLE],
    IsParameterOHO[_BILEVEL_LEARNABLE],
    IsHyperParameterOHO[_BILEVEL_LEARNABLE],
    IsRecurrentWeightsOHO[_BILEVEL_LEARNABLE],
    IsReadoutWeightsOHO[_BILEVEL_LEARNABLE],
    IsLossOHO[_BILEVEL_LEARNABLE],
    IsGradientOHO[_BILEVEL_LEARNABLE],
    IsPredictionOHO[_BILEVEL_LEARNABLE],
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
    IsLossOHO[_OHO_FUTURE],
    IsGradientOHO[_OHO_FUTURE],
    IsPredictionOHO[_OHO_FUTURE],
):
    pass


# # Need to endow on tuple bc input needs to be vectorized to handle batch
# class TrainingInputOhoDataInterpreter(HasTrainingInput[tuple[A, B, C, D], A]):
#     def getTrainingInput(data):
#         trainingInput, _, _, _ = data
#         return trainingInput

# class TrainingLabelOhoDataInterpreter(HasTrainingLabel[tuple[A, B, C, D], B]):
#     def getTrainingLabel(data):
#         _, trainingLabel, _, _ = data
#         return trainingLabel

# class ValidationInputOhoDataInterpreter(HasValidationInput[tuple[A, B, C, D], C]):
#     def getValidationInput(data):
#         _, _, validationInput, _ = data
#         return validationInput

# class ValidationLabelOhoDataInterpreter(HasValidationLabel[tuple[A, B, C, D], D]):
#     def getValidationLabel(data):
#         _, _, _, validationLabel = data
#         return validationLabel


# class OhoStateInterpreter(ActivationOhoStateInterpreter, ParameterOhoStateInterpreter, HyperParameterOhoStateInterpreter, TrainingInputOhoDataInterpreter, TrainingLabelOhoDataInterpreter, ValidationInputOhoDataInterpreter, ValidationLabelOhoDataInterpreter):
#     pass


# test: RnnBPTTState = RnnBPTTState(
#     activation=ACTIVATION(torch.tensor([1.0])),
#     parameter=PARAMETER(torch.tensor([1.0])),
#     trainPrediction=PREDICTION(torch.tensor([1.0])),
#     trainLoss=LOSS(torch.tensor([1.0])),
#     trainGradient=GRADIENT(torch.tensor([1.0])),
#     hyperparameter=HYPERPARAMETER(torch.tensor([1.0])),
#     n_h=1,
#     n_in=1,
#     n_out=1,
#     alpha=1.0,
#     activationFn=lambda x: x)

# test2 = RnnFutureState(
#     activation=ACTIVATION(torch.tensor([1.0])),
#     parameter=PARAMETER(torch.tensor([1.0])),
#     trainPrediction=PREDICTION(torch.tensor([1.0])),
#     trainLoss=LOSS(torch.tensor([1.0])),
#     trainGradient=GRADIENT(torch.tensor([1.0])),
#     hyperparameter=HYPERPARAMETER(torch.tensor([1.0])),
#     n_h=1,
#     n_in=1,
#     n_out=1,
#     alpha=1.0,
#     activationFn=lambda x: x,
#     influenceTensor=INFLUENCETENSOR(torch.tensor([1.0])))


# ACTIV = TypeVar('ACTIV')
# PRED = TypeVar('PRED')
# PARAM = TypeVar('PARAM')
# HPARAM = TypeVar('HPARAM')
# PARAM_T = TypeVar('PARAM_T')
# PARAM_R = TypeVar('PARAM_R')
# PARAM_O = TypeVar('PARAM_O')

# class _Activation(HasActivation[ENV, ACTIV], HasRecurrentWeights[ENV, T, _E], HasParameter[ENV, T], Protocol[ENV, ACTIV, T, _E]):
#     pass

# def testFn(dialect: _Activation[ENV, ACTIV, T, E], env: ENV) -> tuple[ACTIV, E]:
#     x = dialect.getActivation(env)
#     y = dialect.getRecurrentWeights(env, dialect.getParameter(env))
#     return (x, y)

# res = testFn(RnnLearnableInterpreter[RnnBPTTState], test)

# print(res)
