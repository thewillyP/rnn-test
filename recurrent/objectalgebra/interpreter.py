from typing import Self

from recurrent.datarecords import Input2Output1
from recurrent.myrecords import RnnGodState
from recurrent.mytypes import *
from recurrent.parameters import RfloConfig
from recurrent.objectalgebra.typeclasses import *
from recurrent.monad import *
from torch import Tensor
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
    GetUORO[RnnGodState[A, B, C], A],
    PutUORO[RnnGodState[A, B, C], A],
):
    type GOD = RnnGodState[A, B, C]

    def getActivation(self):
        return get().fmap(lambda e: e.activation)

    def putActivation[D](self, s: ACTIVATION) -> Fold[Self, D, GOD, Unit]:
        return (
            ProxyS[RnnGodState[A, B, C]]
            .get()
            .flat_map(lambda e: put(e._replace(activation=s)))
        )

    def getParameter[D](self) -> Fold[Self, D, GOD, A]:
        return get().fmap(lambda e: e.parameter)

    def putParameter[D](self, s: A) -> Fold[Self, D, GOD, Unit]:
        return (
            ProxyS[RnnGodState[A, B, C]]
            .get()
            .flat_map(lambda e: put(e._replace(parameter=s)))
        )

    def getHyperParameter[D](self) -> Fold[Self, D, GOD, B]:
        return get().fmap(lambda e: e.hyperparameter)

    def putHyperParameter[D](self, s: B) -> Fold[Self, D, GOD, Unit]:
        return (
            ProxyS[RnnGodState[A, B, C]]
            .get()
            .flat_map(lambda e: put(e._replace(hyperparameter=s)))
        )

    def getInfluenceTensor[D](self) -> Fold[Self, D, GOD, Gradient[A]]:
        return get().fmap(lambda e: e.influenceTensor)

    def putInfluenceTensor[D](self, s: Gradient[A]) -> Fold[Self, D, GOD, Unit]:
        return (
            ProxyS[RnnGodState[A, B, C]]
            .get()
            .flat_map(lambda e: put(e._replace(influenceTensor=s)))
        )

    def getRfloConfig[D](self) -> Fold[Self, D, GOD, RfloConfig]:
        return get().fmap(lambda e: e.rfloConfig)

    def getRnnConfig[D](self) -> Fold[Self, D, GOD, RnnConfig]:
        return get().fmap(lambda e: e.rnnConfig)

    def getUORO[D](self) -> Fold[Self, D, GOD, UORO_Param[A]]:
        return get().fmap(lambda e: e.uoro)

    def putUORO[D](self, s: UORO_Param[A]) -> Fold[Self, D, GOD, Unit]:
        return (
            ProxyS[RnnGodState[A, B, C]]
            .get()
            .flat_map(lambda e: put(e._replace(uoro=s)))
        )


class DataInterpreter(
    HasInput[Input2Output1, torch.Tensor],
    HasLabel[Input2Output1, torch.Tensor],
    HasPredictionInput[Input2Output1, torch.Tensor],
):
    _prediction_input = torch.empty(0)

    def getInput[E](self) -> Fold[Self, Input2Output1, E, Tensor]:
        return ask().fmap(lambda e: e.x)

    def getLabel[E](self) -> Fold[Self, Input2Output1, E, Tensor]:
        return ask().fmap(lambda e: e.y)

    def getPredictionInput[E](self) -> Fold[Self, Input2Output1, E, Tensor]:
        return pure(self._prediction_input)


# use this as my default interpreter
class BaseRnnInterpreter[A, B, C](BaseRnnGodInterpreter[A, B, C], DataInterpreter): ...


class _Dialect[E, A, B](
    GetParameter[E, A],
    PutParameter[E, A],
    GetHyperParameter[E, B],
    PutHyperParameter[E, B],
):
    pass


# todo: add a data interpreter for this guy later
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
        return get().fmap(lambda e: e.metaHyperparameter)

    def putHyperParameter[D](self, s: C) -> Fold[Self, D, GOD, Unit]:
        type GOD = RnnGodState[A, B, C]
        return (
            ProxyS[RnnGodState[A, B, C]]
            .get()
            .flat_map(lambda e: put(e._replace(metaHyperparameter=s)))
        )

    def getInfluenceTensor[D](self) -> Fold[Self, D, GOD, Gradient[B]]:
        type GOD = RnnGodState[A, B, C]
        return get().fmap(lambda e: e.ohoInfluenceTensor)

    def putInfluenceTensor[D](self, s: Gradient[B]) -> Fold[Self, D, GOD, Unit]:
        type GOD = RnnGodState[A, B, C]
        return (
            ProxyS[RnnGodState[A, B, C]]
            .get()
            .flat_map(lambda e: put(e._replace(ohoInfluenceTensor=s)))
        )

    def getRfloConfig[D](self) -> Fold[Self, D, GOD, RfloConfig]:
        type GOD = RnnGodState[A, B, C]
        return get().fmap(lambda e: e.rfloConfig_bilevel)

    def getRnnConfig[D](self) -> Fold[Self, D, GOD, RnnConfig]:
        type GOD = RnnGodState[A, B, C]
        return get().fmap(lambda e: e.rnnConfig_bilevel)
