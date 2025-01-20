from typing import Generic, Self
from recurrent.mytypes import *
from recurrent.parameters import (
    RfloConfig,
)
from typing import Protocol
from recurrent.monad import *


# ============== Typeclasses ==============
class GetActivation[E, T](Protocol):
    def getActivation[D](self) -> Fold[Self, D, E, T]: ...


class PutActivation[E, T](Protocol):
    def putActivation[D](self, s: T) -> Fold[Self, D, E, Unit]: ...


class GetParameter[E, T](Protocol):
    def getParameter[D](self) -> Fold[Self, D, E, T]: ...


class PutParameter[E, T](Protocol):
    def putParameter[D](self, s: T) -> Fold[Self, D, E, Unit]: ...


class GetHyperParameter[E, T](Protocol):
    def getHyperParameter[D](self) -> Fold[Self, D, E, T]: ...


class PutHyperParameter[E, T](Protocol):
    def putHyperParameter[D](self, s: T) -> Fold[Self, D, E, Unit]: ...


class GetInfluenceTensor[E, T](Protocol):
    def getInfluenceTensor[D](self) -> Fold[Self, D, E, T]: ...


class PutInfluenceTensor[E, T](Protocol):
    def putInfluenceTensor[D](self, s: T) -> Fold[Self, D, E, Unit]: ...


class GetRfloConfig[E](Protocol):
    def getRfloConfig[D](self) -> Fold[Self, D, E, RfloConfig]: ...


class HasInput[D, T](Protocol):
    def getInput[E](self) -> Fold[Self, D, E, T]: ...


class HasPredictionInput[D, T](Protocol):
    def getPredictionInput[E](self) -> Fold[Self, D, E, T]: ...


class HasLabel[D, T](Protocol):
    def getLabel[E](self) -> Fold[Self, D, E, T]: ...
