from typing import Self

from donotation import do

from recurrent.datarecords import InputOutput, OhoInputOutput
from recurrent.myfunc import flipTuple
from recurrent.myrecords import RnnGodState
from recurrent.mytypes import *
from recurrent.parameters import Logs
from recurrent.objectalgebra.typeclasses import *
from recurrent.monad import *
import jax
from jax import Array
import jax.numpy as jnp
import equinox as eqx

from recurrent.util import prng_split

type GOD[A, B, C] = RnnGodState[A, B, C]


class BaseRnnGodInterpreter[A, B, C](
    GetActivation[GOD[A, B, C], ACTIVATION],
    PutActivation[GOD[A, B, C], ACTIVATION],
    GetParameter[GOD[A, B, C], A],
    PutParameter[GOD[A, B, C], A],
    GetHyperParameter[GOD[A, B, C], B],
    PutHyperParameter[GOD[A, B, C], B],
    GetInfluenceTensor[GOD[A, B, C], Jacobian[A]],
    PutInfluenceTensor[GOD[A, B, C], Jacobian[A]],
    GetRnnConfig[GOD[A, B, C]],
    GetUORO[GOD[A, B, C]],
    PutUORO[GOD[A, B, C]],
    PutLog[GOD[A, B, C], Logs],
    GetLog[GOD[A, B, C], Logs],
):
    type God = GOD[A, B, C]

    def getActivation[D](self) -> Fold[Self, D, God, ACTIVATION]:
        return gets(lambda e: e.activation)

    def putActivation[D](self, s: ACTIVATION) -> Fold[Self, D, God, Unit]:
        return modifies(lambda e: eqx.tree_at(lambda t: t.activation, e, s))

    def getParameter[D](self) -> Fold[Self, D, God, A]:
        return gets(lambda e: e.parameter)

    def putParameter[D](self, s: A) -> Fold[Self, D, God, Unit]:
        return modifies(lambda e: eqx.tree_at(lambda t: t.parameter, e, s))

    def getHyperParameter[D](self) -> Fold[Self, D, God, B]:
        return gets(lambda e: e.hyperparameter)

    def putHyperParameter[D](self, s: B) -> Fold[Self, D, God, Unit]:
        return modifies(lambda e: eqx.tree_at(lambda t: t.hyperparameter, e, s))

    def getInfluenceTensor[D](self) -> Fold[Self, D, God, Jacobian[A]]:
        return gets(lambda e: e.influenceTensor)

    def putInfluenceTensor[D](self, s: Jacobian[A]) -> Fold[Self, D, God, Unit]:
        return modifies(lambda e: eqx.tree_at(lambda t: t.influenceTensor, e, s))

    def getRnnConfig[D](self) -> Fold[Self, D, God, RnnConfig]:
        return gets(lambda e: e.rnnConfig)

    def getUORO[D](self) -> Fold[Self, D, God, UORO_Param]:
        return gets(lambda e: e.uoro)

    def putUORO[D](self, s: UORO_Param) -> Fold[Self, D, God, Unit]:
        return modifies(lambda e: eqx.tree_at(lambda t: t.uoro, e, s))

    def putLog[D](self, s: Logs) -> Fold[Self, D, GOD[A, B, C], Unit]:
        return modifies(lambda e: eqx.tree_at(lambda t: t.logs, e, eqx.combine(s, e.logs)))

    def getLog[D](self) -> Fold[Self, D, GOD[A, B, C], Logs]:
        return gets(lambda e: e.logs)


class DataInterpreter(
    HasInput[InputOutput, jax.Array],
    HasLabel[InputOutput, jax.Array],
):
    def getInput[E](self) -> Fold[Self, InputOutput, E, jax.Array]:
        return asks(lambda e: e.x)

    def getLabel[E](self) -> Fold[Self, InputOutput, E, jax.Array]:
        return asks(lambda e: e.y)


class RNGInterpreter[A, B, C](HasPRNG[GOD[A, B, C], PRNG]):
    type God = GOD[A, B, C]

    def generatePRNG[D](self) -> Fold[Self, D, GOD[A, B, C], tuple[PRNG, PRNG]]:
        return gets(lambda e: prng_split(e.prng))

    def putPRNG[D](self, s) -> Fold[Self, D, GOD[A, B, C], Unit]:
        return modifies(lambda e: eqx.tree_at(lambda t: t.prng, e, s))


# use this as my default interpreter
class BaseRnnInterpreter[A, B, C](BaseRnnGodInterpreter[A, B, C], DataInterpreter, RNGInterpreter[A, B, C]): ...


class _Dialect[E, A, B](
    GetParameter[E, A],
    PutParameter[E, A],
    GetHyperParameter[E, B],
    PutHyperParameter[E, B],
):
    pass


# todo: add a data interpreter for this guy later
class BilevelRnnGodInterpreter[A, B, C](
    GetActivation[GOD[A, B, C], A],
    PutActivation[GOD[A, B, C], A],
    GetParameter[GOD[A, B, C], B],
    PutParameter[GOD[A, B, C], B],
    GetHyperParameter[GOD[A, B, C], C],
    PutHyperParameter[GOD[A, B, C], C],
    GetInfluenceTensor[GOD[A, B, C], Gradient[B]],
    PutInfluenceTensor[GOD[A, B, C], Gradient[B]],
    PutLog[GOD[A, B, C], Logs],
    GetLog[GOD[A, B, C], Logs],
):
    type G = GOD[A, B, C]

    def __init__(self, dialect: _Dialect[G, A, B]) -> None:
        super().__init__()
        self.dialect = dialect

    def getActivation[D](self) -> Fold[Self, D, G, A]:
        return self.dialect.getParameter().switch_dl(self.dialect)

    def putActivation[D](self, s: A) -> Fold[Self, D, G, Unit]:
        return self.dialect.putParameter(s).switch_dl(self.dialect)

    def getParameter[D](self) -> Fold[Self, D, G, B]:
        return self.dialect.getHyperParameter().switch_dl(self.dialect)

    def putParameter[D](self, s: B) -> Fold[Self, D, G, Unit]:
        return self.dialect.putHyperParameter(s).switch_dl(self.dialect)

    def getHyperParameter[D](self) -> Fold[Self, D, G, C]:
        return gets(lambda e: e.metaHyperparameter)

    def putHyperParameter[D](self, s: C) -> Fold[Self, D, G, Unit]:
        return modifies(lambda e: eqx.tree_at(lambda t: t.metaHyperParameter, e, s))

    def getInfluenceTensor[D](self) -> Fold[Self, D, G, Gradient[B]]:
        return gets(lambda e: e.ohoInfluenceTensor)

    def putInfluenceTensor[D](self, s: Gradient[B]) -> Fold[Self, D, G, Unit]:
        return modifies(lambda e: eqx.tree_at(lambda t: t.ohoInfluenceTensor, e, s))

    def getRnnConfig[D](self) -> Fold[Self, D, G, RnnConfig]:
        return gets(lambda e: e.rnnConfig_bilevel)

    def putLog[D](self, s: Logs) -> Fold[Self, D, GOD[A, B, C], Unit]:
        return modifies(lambda e: eqx.tree_at(lambda t: t.oho_logs, e, eqx.combine(s, e.oho_logs)))

    def getLog[D](self) -> Fold[Self, D, GOD[A, B, C], Logs]:
        return gets(lambda e: e.oho_logs)


# will end up duping labels w/o using since data oriented for supervised stream
class OhoDataInterpreter(
    HasInput[OhoInputOutput, Traversable[InputOutput]],
    HasPredictionInput[OhoInputOutput, Traversable[InputOutput]],
    HasLabel[OhoInputOutput, Traversable[jax.Array]],
):
    def getInput[E](self) -> Fold[Self, OhoInputOutput, E, Traversable[InputOutput]]:
        return asks(lambda e: e.train)

    def getLabel[E](self) -> Fold[Self, OhoInputOutput, E, Traversable[Array]]:
        return asks(lambda e: e.labels)

    def getPredictionInput[E](self) -> Fold[Self, OhoInputOutput, E, Traversable[InputOutput]]:
        return asks(lambda e: e.validation)


class OhoInterpreter[A, B, C](BilevelRnnGodInterpreter[A, B, C], RNGInterpreter[A, B, C], OhoDataInterpreter): ...
