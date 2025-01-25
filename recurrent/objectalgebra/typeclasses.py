from typing import Generic, Self
from recurrent.mytypes import *
from recurrent.parameters import (
    RfloConfig,
    RnnConfig,
    UORO_Param,
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


class GetRnnConfig[E](Protocol):
    def getRnnConfig[D](self) -> Fold[Self, D, E, RnnConfig]: ...


class HasInput[D, T](Protocol):
    def getInput[E](self) -> Fold[Self, D, E, T]: ...


class HasPredictionInput[D, T](Protocol):
    def getPredictionInput[E](self) -> Fold[Self, D, E, T]: ...


class HasLabel[D, T](Protocol):
    def getLabel[E](self) -> Fold[Self, D, E, T]: ...


class GetUORO[E, Pr](Protocol):
    def getUORO[D](self) -> Fold[Self, D, E, UORO_Param[Pr]]: ...


class PutUORO[E, Pr](Protocol):
    def putUORO[D](self, s: UORO_Param[Pr]) -> Fold[Self, D, E, Unit]: ...


class HasPRNG[E, T](Protocol):
    def generatePRNG[D](self) -> Fold[Self, D, E, tuple[T, T]]: ...

    def putPRNG[D](self, s: T) -> Fold[Self, D, E, Unit]: ...

    @do()
    def updatePRNG[D](self) -> G[Fold[Self, D, E, T]]:
        prng, new_prng = yield from self.generatePRNG()
        _ = yield from self.putPRNG(new_prng)
        return pure(prng)
