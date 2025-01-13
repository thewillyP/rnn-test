from typing import Generic
from recurrent.mytypes import *
from recurrent.parameters import (
    RfloConfig,
)
from typing import Protocol

_ENV = TypeVar("_ENV", contravariant=True)
_DATA = TypeVar("_DATA", contravariant=True)
_T = TypeVar("_T", contravariant=True)
_E = TypeVar("_E", covariant=True)


# ============== Typeclasses ==============
class GetActivation(Generic[_ENV, _E], Protocol):
    def getActivation(self, env: _ENV) -> _E:
        pass


class PutActivation(Generic[ENV, _T], Protocol):
    def putActivation(self, s: _T, env: ENV) -> ENV:
        pass


class GetParameter(Generic[_ENV, _E], Protocol):
    def getParameter(self, env: _ENV) -> _E:
        pass


class PutParameter(Generic[ENV, _T], Protocol):
    def putParameter(self, s: _T, env: ENV) -> ENV:
        pass


class GetHyperParameter(Generic[_ENV, _E], Protocol):
    def getHyperParameter(self, env: _ENV) -> _E:
        pass


class PutHyperParameter(Generic[ENV, _T], Protocol):
    def putHyperParameter(self, s: _T, env: ENV) -> ENV:
        pass


class GetInfluenceTensor(Generic[_ENV, _E], Protocol):
    def getInfluenceTensor(self, env: _ENV) -> _E:
        pass


class PutInfluenceTensor(Generic[ENV, _T], Protocol):
    def putInfluenceTensor(self, s: _T, env: ENV) -> ENV:
        pass


class GetRfloConfig(Generic[_ENV], Protocol):
    def getRfloConfig(self, env: _ENV) -> RfloConfig:
        pass


class HasInput(Generic[_DATA, _E], Protocol):
    def getInput(self, data: _DATA) -> _E:
        pass


class HasPredictionInput(Generic[_DATA, _E], Protocol):
    def getPredictionInput(self, data: _DATA) -> _E:
        pass


class HasLabel(Generic[_DATA, _E], Protocol):
    def getLabel(self, data: _DATA) -> _E:
        pass
