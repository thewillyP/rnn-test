from typing import Generic
from recurrent.mytypes import *
from recurrent.myrecords import (
    RfloConfig,
)
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


# class GetLoss(Generic[_ENV, _E], Protocol):
#     @staticmethod
#     def getLoss(env: _ENV) -> _E:
#         pass


# class PutLoss(Generic[ENV, _T], Protocol):
#     @staticmethod
#     def putLoss(s: _T, env: ENV) -> ENV:
#         pass


# class GetGradient(Generic[_ENV, _E], Protocol):
#     @staticmethod
#     def getGradient(env: _ENV) -> _E:
#         pass


# class PutGradient(Generic[ENV, _T], Protocol):
#     @staticmethod
#     def putGradient(s: _T, env: ENV) -> ENV:
#         pass


# class GetPrediction(Generic[_ENV, _E], Protocol):
#     @staticmethod
#     def getPrediction(env: _ENV) -> _E:
#         pass


# class PutPrediction(Generic[ENV, _T], Protocol):
#     @staticmethod
#     def putPrediction(s: _T, env: ENV) -> ENV:
#         pass


class GetRfloConfig(Generic[_ENV], Protocol):
    @staticmethod
    def getRfloConfig(env: _ENV) -> RfloConfig:
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
