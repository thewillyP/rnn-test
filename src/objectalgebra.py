from abc import ABCMeta, abstractmethod
from typing import TypeVar, Generic
from dataclasses import dataclass

T = TypeVar('T')
E = TypeVar('E')
A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')
D = TypeVar('D')
X = TypeVar('X')
Y = TypeVar('Y')
H = TypeVar('H')
P = TypeVar('P')
L = TypeVar('L')
HP = TypeVar('HP')

ENV = TypeVar('ENV')

DATA = TypeVar('DATA')

# ============== Typeclasses ==============
class HasActivation(Generic[ENV, T], metaclass=ABCMeta):
    @abstractmethod
    def getActivation(self, env: ENV) -> T:
        pass

    @abstractmethod
    def putActivation(self, s: T, env: ENV) -> ENV:
        pass


class HasParameter(Generic[ENV, T], metaclass=ABCMeta):
    @abstractmethod
    def getParameter(self, env: ENV) -> T:
        pass

    @abstractmethod
    def putParameter(self, s: T, env: ENV) -> ENV:
        pass

class HasHyperParameter(Generic[ENV, T], metaclass=ABCMeta):
    @abstractmethod
    def getHyperParameter(self, env: ENV) -> T:
        pass

    @abstractmethod
    def putHyperParameter(self, s: T, env: ENV) -> ENV:
        pass

class HasTrainingInput(Generic[DATA, T], metaclass=ABCMeta):
    @abstractmethod
    def getTrainingInput(self, data: DATA) -> T:
        pass

class HasTrainingLabel(Generic[DATA, T], metaclass=ABCMeta):
    @abstractmethod
    def getTrainingLabel(self, data: DATA) -> T:
        pass

class HasValidationInput(Generic[DATA, T], metaclass=ABCMeta):
    @abstractmethod
    def getValidationInput(self, data: DATA) -> T:
        pass

class HasValidationLabel(Generic[DATA, T], metaclass=ABCMeta):
    @abstractmethod
    def getValidationLabel(self, data: DATA) -> T:
        pass


# ============== Instance ==============

# downside is that I need to create a whole new class that inherits all the instantiations. 
# I will just maintain and reedit my god instance interpreter. Best compromise. Not truly extensible but I have full control. 

@dataclass(frozen=True)
class OhoState(Generic[A, B, C]): 
    activation: A
    parameter: B
    hyperparameter: C

    
class ActivationOhoStateInterpreter(HasActivation[OhoState[A, B, C], A]):
    def getActivation(self, env):
        return env.activation
    
    def putActivation(self, s, env):
        return OhoState(s, env.parameter, env.hyperparameter)
    
class ParameterOhoStateInterpreter(HasParameter[OhoState[A, B, C], B]):
    def getParameter(self, env):
        return env.parameter
    
    def putParameter(self, s, env):
        return OhoState(env.activation, s, env.hyperparameter)
    
class HyperParameterOhoStateInterpreter(HasHyperParameter[OhoState[A, B, C], C]):
    def getHyperParameter(self, env):
        return env.hyperparameter
    
    def putHyperParameter(self, s, env):
        return OhoState(env.activation, env.parameter, s)

# Need to endow on tuple bc input needs to be vectorized to handle batch
class TrainingInputOhoDataInterpreter(HasTrainingInput[tuple[A, B, C, D], A]):
    def getTrainingInput(self, data):
        trainingInput, _, _, _ = data
        return trainingInput

class TrainingLabelOhoDataInterpreter(HasTrainingLabel[tuple[A, B, C, D], B]):
    def getTrainingLabel(self, data):
        _, trainingLabel, _, _ = data
        return trainingLabel

class ValidationInputOhoDataInterpreter(HasValidationInput[tuple[A, B, C, D], C]):
    def getValidationInput(self, data):
        _, _, validationInput, _ = data
        return validationInput

class ValidationLabelOhoDataInterpreter(HasValidationLabel[tuple[A, B, C, D], D]):
    def getValidationLabel(self, data):
        _, _, _, validationLabel = data
        return validationLabel

# class TestInputOhoDataInterpreter(HasTrainingInput[tuple[A, B], A]):
#     def getTrainingInput(self, data):
#         trainingInput, _ = data
#         return trainingInput

# class TestLabelOhoDataInterpreter(HasTrainingLabel[tuple[A, B], B]):
#     def getTrainingLabel(self, data):
#         _, trainingLabel = data
#         return trainingLabel

class OhoStateInterpreter(ActivationOhoStateInterpreter, ParameterOhoStateInterpreter, HyperParameterOhoStateInterpreter, TrainingInputOhoDataInterpreter, TrainingLabelOhoDataInterpreter, ValidationInputOhoDataInterpreter, ValidationLabelOhoDataInterpreter):
    pass

# class OhoStateInterpreterTest(ActivationOhoStateInterpreter, TestInputOhoDataInterpreter, TestLabelOhoDataInterpreter):
#     pass