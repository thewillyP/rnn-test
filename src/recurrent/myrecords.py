from dataclasses import dataclass
from donotation import do
from recurrent.monad import *
from recurrent.mytypes import *
from recurrent.parameters import *
from recurrent.util import prng_split


type MlApp[Data, A, B, X] = App[Interpreter[Data, A, B], Data, GodState[Data, A, B], X]


@dataclass(frozen=True)
class GodState[Data, A, B](eqx.Module):
    states: list["State[Data, A, B]"]
    prng: PRNG


@dataclass(frozen=True)
class State(eqx.Module):
    activation: ACTIVATION
    influenceTensor: JACOBIAN
    rnnParameter: RnnParameter
    sgdParameter: SgdParameter
    uoro: UORO_Param
    logs: Logs
    rnnConfig: RnnConfig = eqx.field(static=True)


@dataclass(frozen=True)
class Interpreter[Data, A, B]:
    type LocalApp[X] = MlApp[Data, A, B, X]

    getReccurentState: LocalApp[A]
    putReccurentState: Callable[[A], LocalApp[Unit]]
    getRecurrentParam: LocalApp[B]
    putRecurrentParam: Callable[[B], LocalApp[Unit]]

    getActivation: LocalApp[ACTIVATION]
    putActivation: Callable[[ACTIVATION], LocalApp[Unit]]
    getInfluenceTensor: LocalApp[JACOBIAN]
    putInfluenceTensor: Callable[[JACOBIAN], LocalApp[Unit]]
    getUoro: LocalApp[UORO_Param]
    putUoro: Callable[[UORO_Param], LocalApp[Unit]]
    getRnnConfig: LocalApp[RnnConfig]
    putLogs: Callable[[Logs], LocalApp[Unit]]

    @do()
    def updatePRNG(self) -> G[LocalApp[PRNG]]:
        prng, new_prng = yield from gets(lambda e: prng_split(e.prng))
        _ = yield from modifies(lambda e: eqx.tree_at(lambda t: t.prng, e, new_prng))
        return pure(prng, PX3())
