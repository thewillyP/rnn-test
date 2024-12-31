from abc import ABCMeta, abstractmethod
from typing import Callable, Generic
from dataclasses import dataclass, replace
from mytypes import *
from myrecords import WithBaseFuture, WithBilevel, WithOhoFuture, RnnLearnable, RnnEnv, BilevelLearnable
from util import rnnSplitParameters
from typing import Protocol

# ============== Typeclasses ==============
class HasActivation(Generic[ENV, T], metaclass=ABCMeta):
    @abstractmethod
    @staticmethod
    def getActivation(env: ENV) -> T:
        pass
    
    @abstractmethod
    @staticmethod
    def putActivation(s: T, env: ENV) -> ENV:
        pass


class HasParameter(Generic[ENV, T], metaclass=ABCMeta):
    @abstractmethod
    @staticmethod
    def getParameter(env: ENV) -> T:
        pass

    @abstractmethod
    @staticmethod
    def putParameter(s: T, env: ENV) -> ENV:
        pass

class HasHyperParameter(Generic[ENV, T], metaclass=ABCMeta):
    @abstractmethod
    @staticmethod
    def getHyperParameter(env: ENV) -> T:
        pass

    @abstractmethod
    @staticmethod
    def putHyperParameter(s: T, env: ENV) -> ENV:
        pass

class HasInfluenceTensor(Generic[ENV, T], metaclass=ABCMeta):
    @abstractmethod
    @staticmethod
    def getInfluenceTensor(env: ENV) -> T:
        pass

    @abstractmethod
    @staticmethod
    def putInfluenceTensor(s: T, env: ENV) -> ENV:
        pass

class HasRecurrentWeights(Generic[ENV, T, E], metaclass=ABCMeta):
    @abstractmethod
    @staticmethod
    def getRecurrentWeights(env: ENV, s: T) -> E:
        pass

class HasReadoutWeights(Generic[ENV, T, E], metaclass=ABCMeta):
    @abstractmethod
    @staticmethod
    def getReadoutWeights(env: ENV, s: T) -> E:
        pass

class HasLoss(Generic[ENV, T], metaclass=ABCMeta):
    @abstractmethod
    @staticmethod
    def getLoss(env: ENV) -> T:
        pass

    @abstractmethod
    @staticmethod
    def putLoss(s: T, env: ENV) -> ENV:
        pass

class HasGradient(Generic[ENV, T], metaclass=ABCMeta):
    @abstractmethod
    @staticmethod
    def getGradient(env: ENV) -> T:
        pass

    @abstractmethod
    @staticmethod
    def putGradient(s: T, env: ENV) -> ENV:
        pass

class HasPrediction(Generic[ENV, T], metaclass=ABCMeta):
    @abstractmethod
    @staticmethod
    def getPrediction(env: ENV) -> T:
        pass

    @abstractmethod
    @staticmethod
    def putPrediction(s: T, env: ENV) -> ENV:
        pass

class HasInput(Generic[DATA, T], metaclass=ABCMeta):
    @abstractmethod
    @staticmethod
    def getInput(data: DATA) -> T:
        pass

class HasPredictionInput(Generic[DATA, T], metaclass=ABCMeta):
    @abstractmethod
    @staticmethod
    def getPredictionInput(data: DATA) -> T:
        pass

class HasLabel(Generic[DATA, T], metaclass=ABCMeta):
    @abstractmethod
    @staticmethod
    def getLabel(data: DATA) -> T:
        pass

# class HasValidationInput(Generic[DATA, T], metaclass=ABCMeta):
#     @abstractmethod
#     @staticmethod
#     def getValidationInput(data: DATA) -> T:
#         pass

# class HasValidationLabel(Generic[DATA, T], metaclass=ABCMeta):
#     @abstractmethod
#     @staticmethod
#     def getValidationLabel(data: DATA) -> T:
#         pass


# ============== Instance ==============

# downside is that I need to create a whole new class that inherits all the instantiations. 
# I will just maintain and reedit my god instance interpreter. Best compromise. Not truly extensible but I have full control. 

# @dataclass(frozen=True)
# class OhoState(Generic[A, B, C]): 
#     activation: A
#     parameter: B
#     hyperparameter: C



class IsActivation(HasActivation[RnnEnv, ACTIVATION]):
    @staticmethod
    def getActivation(env: RnnEnv) -> ACTIVATION:
        return env.activation

    @staticmethod
    def putActivation(env: RnnEnv, s: ACTIVATION) -> RnnEnv:
        return replace(env, activation=s)

class IsParameter(HasParameter[RnnEnv, PARAMETER]):
    @staticmethod
    def getParameter(env: RnnEnv) -> PARAMETER:
        return env.parameter

    @staticmethod
    def putParameter(env: RnnEnv, s: PARAMETER) -> RnnEnv:
        return replace(env, parameter=s)

class IsRecurrentWeights(HasRecurrentWeights[RnnEnv, PARAMETER, PARAMETER]):
    @staticmethod
    def getRecurrentWeights(env: RnnEnv, s: PARAMETER) -> PARAMETER:
        wrec, _ = rnnSplitParameters(env, s)
        return wrec

class IsReadoutWeights(HasReadoutWeights[RnnEnv, PARAMETER, PARAMETER]):
    @staticmethod
    def getReadoutWeights(env: RnnEnv, s: PARAMETER) -> PARAMETER:
        _, wout = rnnSplitParameters(env, s)
        return wout

class IsHyperParameter(HasHyperParameter[RnnLearnable, HYPERPARAMETER]):
    @staticmethod
    def getHyperParameter(env: RnnLearnable) -> HYPERPARAMETER:
        return env.hyperparameter

    @staticmethod
    def putHyperParameter(env: RnnLearnable, s: HYPERPARAMETER) -> RnnLearnable:
        return replace(env, hyperparameter=s)

class IsLoss(HasLoss[RnnLearnable, LOSS]):
    @staticmethod
    def getLoss(env: RnnLearnable) -> LOSS:
        return env.trainLoss

    @staticmethod
    def putLoss(env: RnnLearnable, s: LOSS) -> RnnLearnable:
        return replace(env, loss=s)

class IsGradient(HasGradient[RnnLearnable, GRADIENT]):
    @staticmethod
    def getGradient(env: RnnLearnable) -> GRADIENT:
        return env.trainGradient

    @staticmethod
    def putGradient(env: RnnLearnable, s: GRADIENT) -> RnnLearnable:
        return replace(env, gradient=s)

class IsPrediction(HasPrediction[RnnLearnable, PREDICTION]):
    @staticmethod
    def getPrediction(env: RnnLearnable) -> PREDICTION:
        return env.trainPrediction

    @staticmethod
    def putPrediction(env: RnnLearnable, s: PREDICTION) -> RnnLearnable:
        return replace(env, prediction=s)

class _BaseFutureCap(RnnLearnable, WithBaseFuture, Protocol): pass

class IsInfluenceTensor(HasInfluenceTensor[_BaseFutureCap, INFLUENCETENSOR]):
    @staticmethod
    def getInfluenceTensor(env: _BaseFutureCap) -> INFLUENCETENSOR:
        return env.influenceTensor

    @staticmethod
    def putInfluenceTensor(env: _BaseFutureCap, s: INFLUENCETENSOR) -> _BaseFutureCap:
        return replace(env, influenceTensor=s)

class RnnInterpreter(IsActivation, IsParameter, IsRecurrentWeights, IsReadoutWeights):
    pass

class RnnLearnableInterpreter(IsActivation, IsParameter, IsHyperParameter, IsRecurrentWeights, IsReadoutWeights, IsLoss, IsGradient, IsPrediction):
    pass

class RnnWithFutureInterpreter(IsActivation, IsParameter, IsHyperParameter, IsRecurrentWeights, IsReadoutWeights, IsInfluenceTensor, IsLoss, IsGradient, IsPrediction):
    pass


class IsActivationOHO(HasActivation[BilevelLearnable, PARAMETER]):
    getActivation = staticmethod(RnnLearnableInterpreter.getParameter)
    putActivation = staticmethod(RnnLearnableInterpreter.putParameter)

class IsParameterOHO(HasParameter[BilevelLearnable, HYPERPARAMETER]):
    getParameter = staticmethod(RnnLearnableInterpreter.getHyperParameter)
    putParameter = staticmethod(RnnLearnableInterpreter.putHyperParameter)

class IsLossOHO(HasLoss[BilevelLearnable, LOSS]):
    @staticmethod
    def getLoss(env: BilevelLearnable) -> LOSS:
        return env.validationLoss
    
    @staticmethod
    def putLoss(s: LOSS, env: BilevelLearnable) -> BilevelLearnable:
        return replace(env, validationLoss=s)

class IsGradientOHO(HasGradient[BilevelLearnable, GRADIENT]):
    @staticmethod
    def getGradient(env: BilevelLearnable) -> GRADIENT:
        return env.validationGradient
    
    @staticmethod
    def putGradient(s: GRADIENT, env: BilevelLearnable) -> BilevelLearnable:
        return replace(env, validationGradient=s)

class IsPredictionOHO(HasPrediction[BilevelLearnable, PREDICTION]):
    @staticmethod
    def getPrediction(env: BilevelLearnable) -> PREDICTION:
        return env.validationPrediction
    
    @staticmethod
    def putPrediction(s: PREDICTION, env: BilevelLearnable) -> BilevelLearnable:
        return replace(env, validationPrediction=s)

class IsHyperParameterOHO(HasHyperParameter[BilevelLearnable, METAHYPERPARAMETER]):
    @staticmethod
    def getHyperParameter(env: BilevelLearnable) -> METAHYPERPARAMETER:
        return env.metaHyperparameter
    
    @staticmethod
    def putHyperParameter(s: METAHYPERPARAMETER, env: BilevelLearnable) -> BilevelLearnable:
        return replace(env, metaHyperparameter=s)

class IsRecurrentWeightsOHO(HasRecurrentWeights[BilevelLearnable, HYPERPARAMETER, HYPERPARAMETER]):
    @staticmethod
    def getRecurrentWeights(env: BilevelLearnable, _: HYPERPARAMETER) -> HYPERPARAMETER:
        return BilevelInterpreter.getParameter(env)
    
class IsReadoutWeightsOHO(HasReadoutWeights[BilevelLearnable, HYPERPARAMETER, HYPERPARAMETER]):
    @staticmethod
    def getReadoutWeights(env: BilevelLearnable, _: HYPERPARAMETER) -> HYPERPARAMETER:
        return BilevelInterpreter.getParameter(env)
    
class _OhoFutureCap(BilevelLearnable, WithOhoFuture, Protocol): pass

class IsInfluenceTensorOHO(HasInfluenceTensor[_OhoFutureCap, INFLUENCETENSOR]):
    @staticmethod
    def getInfluenceTensor(env: _OhoFutureCap) -> INFLUENCETENSOR:
        return env.ohoInfluenceTensor

    @staticmethod
    def putInfluenceTensor(s: INFLUENCETENSOR, env: _OhoFutureCap) -> _OhoFutureCap:
        return replace(env, ohoInfluenceTensor=s)

class BilevelInterpreter(IsActivationOHO, IsParameterOHO, IsHyperParameterOHO, IsRecurrentWeightsOHO, IsReadoutWeightsOHO, IsLossOHO, IsGradientOHO, IsPredictionOHO):
    pass

class BilevelWithOhoInterpreter(IsActivationOHO, IsParameterOHO, IsHyperParameterOHO, IsRecurrentWeightsOHO, IsReadoutWeightsOHO, IsInfluenceTensorOHO, IsLossOHO, IsGradientOHO, IsPredictionOHO):
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








