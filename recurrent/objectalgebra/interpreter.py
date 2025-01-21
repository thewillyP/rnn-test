from typing import Self

from recurrent.myrecords import RnnGodState
from recurrent.mytypes import *
from recurrent.parameters import RfloConfig
from recurrent.objectalgebra.typeclasses import *
from recurrent.monad import *
import copy


class BaseRnnGodInterpreter[A, B, C](
    GetActivation[RnnGodState[A, B, C], ACTIVATION],
    PutActivation[RnnGodState[A, B, C], ACTIVATION],
    GetParameter[RnnGodState[A, B, C], A],
    PutParameter[RnnGodState[A, B, C], A],
    GetHyperParameter[RnnGodState[A, B, C], B],
    PutHyperParameter[RnnGodState[A, B, C], B],
    GetInfluenceTensor[RnnGodState[A, B, C], Gradient[A]],
    PutInfluenceTensor[RnnGodState[A, B, C], Gradient[A]],
    GetRfloConfig[RnnGodState[A, B, C]],
    GetRnnConfig[RnnGodState[A, B, C]],
):
    type GOD = RnnGodState[A, B, C]

    def getActivation[D](self) -> Fold[Self, D, GOD, ACTIVATION]:
        type GOD = RnnGodState[A, B, C]
        return get(Proxy[GOD]()).fmap(lambda e: e.activation)

    def putActivation[D](self, s: ACTIVATION) -> Fold[Self, D, GOD, Unit]:
        type GOD = RnnGodState[A, B, C]
        return get(Proxy[GOD]()).flat_map(lambda e: put(copy.replace(e, activation=s)))

    def getParameter[D](self) -> Fold[Self, D, GOD, A]:
        type GOD = RnnGodState[A, B, C]
        return get(Proxy[GOD]()).fmap(lambda e: e.parameter)

    def putParameter[D](self, s: A) -> Fold[Self, D, GOD, Unit]:
        type GOD = RnnGodState[A, B, C]
        return get(Proxy[GOD]()).flat_map(lambda e: put(copy.replace(e, parameter=s)))

    def getHyperParameter[D](self) -> Fold[Self, D, GOD, B]:
        type GOD = RnnGodState[A, B, C]
        return get(Proxy[GOD]()).fmap(lambda e: e.hyperparameter)

    def putHyperParameter[D](self, s: B) -> Fold[Self, D, GOD, Unit]:
        type GOD = RnnGodState[A, B, C]
        return get(Proxy[GOD]()).flat_map(
            lambda e: put(copy.replace(e, hyperparameter=s))
        )

    def getInfluenceTensor[D](self) -> Fold[Self, D, GOD, Gradient[A]]:
        type GOD = RnnGodState[A, B, C]
        return get(Proxy[GOD]()).fmap(lambda e: e.influenceTensor)

    def putInfluenceTensor[D](self, s: Gradient[A]) -> Fold[Self, D, GOD, Unit]:
        type GOD = RnnGodState[A, B, C]
        return get(Proxy[GOD]()).flat_map(
            lambda e: put(copy.replace(e, influenceTensor=s))
        )

    def getRfloConfig[D](self) -> Fold[Self, D, GOD, RfloConfig]:
        type GOD = RnnGodState[A, B, C]
        return get(Proxy[GOD]()).fmap(lambda e: e.rfloConfig)

    def getRnnConfig[D](self) -> Fold[Self, D, GOD, RnnConfig]:
        type GOD = RnnGodState[A, B, C]
        return get(Proxy[GOD]()).fmap(lambda e: e.rnnConfig)


class _Dialect[E, A, B](
    GetParameter[E, A],
    PutParameter[E, A],
    GetHyperParameter[E, B],
    PutHyperParameter[E, B],
):
    pass


class BilevelRnnGodInterpreter[A, B, C](
    GetActivation[RnnGodState[A, B, C], A],
    PutActivation[RnnGodState[A, B, C], A],
    GetParameter[RnnGodState[A, B, C], B],
    PutParameter[RnnGodState[A, B, C], B],
    GetHyperParameter[RnnGodState[A, B, C], C],
    PutHyperParameter[RnnGodState[A, B, C], C],
    GetInfluenceTensor[RnnGodState[A, B, C], Gradient[B]],
    PutInfluenceTensor[RnnGodState[A, B, C], Gradient[B]],
    GetRfloConfig[RnnGodState[A, B, C]],
):
    type GOD = RnnGodState[A, B, C]

    def __init__(self, dialect: _Dialect[RnnGodState[A, B, C], A, B]) -> None:
        super().__init__()
        self.dialect = dialect

    def getActivation[D](self) -> Fold[Self, D, GOD, A]:
        return self.dialect.getParameter().switch_dl(self.dialect)

    def putActivation[D](self, s: A) -> Fold[Self, D, GOD, Unit]:
        return self.dialect.putParameter(s).switch_dl(self.dialect)

    def getParameter[D](self) -> Fold[Self, D, GOD, B]:
        return self.dialect.getHyperParameter().switch_dl(self.dialect)

    def putParameter[D](self, s: B) -> Fold[Self, D, GOD, Unit]:
        return self.dialect.putHyperParameter(s).switch_dl(self.dialect)

    def getHyperParameter[D](self) -> Fold[Self, D, GOD, C]:
        type GOD = RnnGodState[A, B, C]
        return get(Proxy[GOD]()).fmap(lambda e: e.metaHyperparameter)

    def putHyperParameter[D](self, s: C) -> Fold[Self, D, GOD, Unit]:
        type GOD = RnnGodState[A, B, C]
        return get(Proxy[GOD]()).flat_map(
            lambda e: put(copy.replace(e, metaHyperparameter=s))
        )

    def getInfluenceTensor[D](self) -> Fold[Self, D, GOD, Gradient[B]]:
        type GOD = RnnGodState[A, B, C]
        return get(Proxy[GOD]()).fmap(lambda e: e.ohoInfluenceTensor)

    def putInfluenceTensor[D](self, s: Gradient[B]) -> Fold[Self, D, GOD, Unit]:
        type GOD = RnnGodState[A, B, C]
        return get(Proxy[GOD]()).flat_map(
            lambda e: put(copy.replace(e, ohoInfluenceTensor=s))
        )

    def getRfloConfig[D](self) -> Fold[Self, D, GOD, RfloConfig]:
        type GOD = RnnGodState[A, B, C]
        return get(Proxy[GOD]()).fmap(lambda e: e.rfloConfig_bilevel)

    def getRnnConfig[D](self) -> Fold[Self, D, GOD, RnnConfig]:
        type GOD = RnnGodState[A, B, C]
        return get(Proxy[GOD]()).fmap(lambda e: e.rnnConfig_bilevel)
