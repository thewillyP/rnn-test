from abc import ABCMeta, abstractmethod
from typing import Callable, Generic
from dataclasses import dataclass, replace
from mytypes import *
from myrecords import RnnGod, WithBaseFuture, WithBilevel, WithOhoFuture
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


class HasTrainingInput(Generic[DATA, T], metaclass=ABCMeta):
    @abstractmethod
    @staticmethod
    def getTrainingInput(data: DATA) -> T:
        pass

class HasTrainingLabel(Generic[DATA, T], metaclass=ABCMeta):
    @abstractmethod
    @staticmethod
    def getTrainingLabel(data: DATA) -> T:
        pass

class HasValidationInput(Generic[DATA, T], metaclass=ABCMeta):
    @abstractmethod
    @staticmethod
    def getValidationInput(data: DATA) -> T:
        pass

class HasValidationLabel(Generic[DATA, T], metaclass=ABCMeta):
    @abstractmethod
    @staticmethod
    def getValidationLabel(data: DATA) -> T:
        pass


# ============== Instance ==============

# downside is that I need to create a whole new class that inherits all the instantiations. 
# I will just maintain and reedit my god instance interpreter. Best compromise. Not truly extensible but I have full control. 

# @dataclass(frozen=True)
# class OhoState(Generic[A, B, C]): 
#     activation: A
#     parameter: B
#     hyperparameter: C



class IsActivation(HasActivation[RnnGod, ACTIVATION]):
    @staticmethod
    def getActivation(env: RnnGod) -> ACTIVATION:
        return env.activation

    @staticmethod
    def putActivation(s: ACTIVATION, env: RnnGod) -> RnnGod:
        return replace(env, activation=s)

class IsParameter(HasParameter[RnnGod, PARAMETER]):
    @staticmethod
    def getParameter(env: RnnGod) -> PARAMETER:
        return env.parameter

    @staticmethod
    def putParameter(s: PARAMETER, env: RnnGod) -> RnnGod:
        return replace(env, parameter=s)

class IsHyperParameter(HasHyperParameter[RnnGod, HYPERPARAMETER]):
    @staticmethod
    def getHyperParameter(env: RnnGod) -> HYPERPARAMETER:
        return env.hyperparameter

    @staticmethod
    def putHyperParameter(s: HYPERPARAMETER, env: RnnGod) -> RnnGod:
        return replace(env, hyperparameter=s)

class IsRecurrentWeights(HasRecurrentWeights[RnnGod, PARAMETER, PARAMETER]):
    @staticmethod
    def getRecurrentWeights(env: RnnGod, s: PARAMETER) -> PARAMETER:
        wrec, _ = rnnSplitParameters(env, s)
        return wrec

class IsReadoutWeights(HasReadoutWeights[RnnGod, PARAMETER, PARAMETER]):
    @staticmethod
    def getReadoutWeights(env: RnnGod, s: PARAMETER) -> PARAMETER:
        _, wout = rnnSplitParameters(env, s)
        return wout

class _BaseFutureCap(RnnGod, WithBaseFuture, Protocol): pass

class IsInfluenceTensor(HasInfluenceTensor[_BaseFutureCap, INFLUENCETENSOR]):
    @staticmethod
    def getInfluenceTensor(env: _BaseFutureCap) -> INFLUENCETENSOR:
        return env.influenceTensor

    @staticmethod
    def putInfluenceTensor(s: INFLUENCETENSOR, env: _BaseFutureCap) -> _BaseFutureCap:
        return replace(env, influenceTensor=s)

class RnnInterpreter(IsActivation, IsParameter, IsHyperParameter, IsRecurrentWeights, IsReadoutWeights):
    pass

class RnnWithFutureInterpreter(IsActivation, IsParameter, IsHyperParameter, IsRecurrentWeights, IsReadoutWeights, IsInfluenceTensor):
    pass

class _BilevelCap(RnnGod, WithBilevel, Protocol): pass

class IsActivationOHO(HasActivation[_BilevelCap, PARAMETER]):
    getActivation = staticmethod(RnnInterpreter.getParameter)
    putActivation = staticmethod(RnnInterpreter.putParameter)

class IsParameterOHO(HasParameter[_BilevelCap, HYPERPARAMETER]):
    getParameter = staticmethod(RnnInterpreter.getHyperParameter)
    putParameter = staticmethod(RnnInterpreter.putHyperParameter)

class IsHyperParameterOHO(HasHyperParameter[_BilevelCap, METAHYPERPARAMETER]):
    @staticmethod
    def getHyperParameter(env: _BilevelCap) -> METAHYPERPARAMETER:
        return env.metaHyperparameter
    
    @staticmethod
    def putHyperParameter(s: METAHYPERPARAMETER, env: _BilevelCap) -> _BilevelCap:
        return replace(env, metaHyperparameter=s)

class IsRecurrentWeightsOHO(HasRecurrentWeights[_BilevelCap, HYPERPARAMETER, HYPERPARAMETER]):
    @staticmethod
    def getRecurrentWeights(env: _BilevelCap, _: HYPERPARAMETER) -> HYPERPARAMETER:
        return BilevelInterpreter.getParameter(env)
    
class IsReadoutWeightsOHO(HasReadoutWeights[_BilevelCap, HYPERPARAMETER, HYPERPARAMETER]):
    @staticmethod
    def getReadoutWeights(env: _BilevelCap, _: HYPERPARAMETER) -> HYPERPARAMETER:
        return BilevelInterpreter.getParameter(env)
    
class _OhoFutureCap(RnnGod, WithBilevel, WithOhoFuture, Protocol): pass

class IsInfluenceTensorOHO(HasInfluenceTensor[_OhoFutureCap, INFLUENCETENSOR]):
    @staticmethod
    def getInfluenceTensor(env: _OhoFutureCap) -> INFLUENCETENSOR:
        return env.ohoInfluenceTensor

    @staticmethod
    def putInfluenceTensor(s: INFLUENCETENSOR, env: _OhoFutureCap) -> _OhoFutureCap:
        return replace(env, ohoInfluenceTensor=s)

class BilevelInterpreter(IsActivationOHO, IsParameterOHO, IsHyperParameterOHO, IsRecurrentWeightsOHO, IsReadoutWeightsOHO):
    pass

class BilevelWithOhoInterpreter(IsActivationOHO, IsParameterOHO, IsHyperParameterOHO, IsRecurrentWeightsOHO, IsReadoutWeightsOHO, IsInfluenceTensorOHO):
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








