from typing import Generic
from dataclasses import dataclass, replace
from recurrent.mytypes import *
from recurrent.mixins import (
    WithBilevelSgdParameter,
    WithOhoPast,
    WithBilevelRflo,
)
from typing import Protocol
from recurrent.objectalgebra.typeclasses import *
from recurrent.parameters import SgdParameter


class _Meta0Dialect(
    Generic[ENV, T, E],
    GetParameter[ENV, T],
    PutParameter[ENV, T],
    GetHyperParameter[ENV, E],
    PutHyperParameter[ENV, E],
):
    pass


class Meta1ActivationParameter(
    Generic[ENV, T, E],
    GetActivation[ENV, T],
    PutActivation[ENV, T],
    GetParameter[ENV, E],
    PutParameter[ENV, E],
):
    def __init__(self, meta0Dialect: _Meta0Dialect[ENV, T, E]) -> None:
        super().__init__()
        self.meta0Dialect = meta0Dialect

    def getActivation(self, env: ENV) -> T:
        return self.meta0Dialect.getParameter(env)

    def putActivation(self, s: T, env: ENV) -> ENV:
        return self.meta0Dialect.putParameter(s, env)

    def getParameter(self, env: ENV) -> E:
        return self.meta0Dialect.getHyperParameter(env)

    def putParameter(self, s: E, env: ENV) -> ENV:
        return self.meta0Dialect.putHyperParameter(s, env)


_BILEVEL_SGD = TypeVar("_BILEVEL_SGD", bound=WithBilevelSgdParameter)


class Meta1Hyperparameter(
    Generic[_BILEVEL_SGD],
    GetHyperParameter[_BILEVEL_SGD, SgdParameter],
    PutHyperParameter[_BILEVEL_SGD, SgdParameter],
):
    def getHyperParameter(self, env: _BILEVEL_SGD) -> SgdParameter:
        return env.metaHyperparameter

    def putHyperParameter(self, s: SgdParameter, env: _BILEVEL_SGD) -> _BILEVEL_SGD:
        return replace(env, metaHyperparameter=s)


_OHO_FUTURE = TypeVar("_OHO_FUTURE", bound=WithOhoPast)


class IsInfluenceTensorOHO(
    Generic[_OHO_FUTURE],
    GetInfluenceTensor[_OHO_FUTURE, INFLUENCETENSOR],
    PutInfluenceTensor[_OHO_FUTURE, INFLUENCETENSOR],
):
    def getInfluenceTensor(self, env: _OHO_FUTURE) -> INFLUENCETENSOR:
        return env.ohoInfluenceTensor

    def putInfluenceTensor(self, s: INFLUENCETENSOR, env: _OHO_FUTURE) -> _OHO_FUTURE:
        return replace(env, ohoInfluenceTensor=s)


_RFLO_ALGEBRA = TypeVar("_RFLO_ALGEBRA", bound=WithBilevelRflo)


class IsRflo(
    Generic[_RFLO_ALGEBRA],
    GetRfloConfig[_RFLO_ALGEBRA],
):
    def getRfloConfig(self, env: _RFLO_ALGEBRA) -> RfloConfig:
        return env.rfloConfig_bilevel


_META1_WITH_SGD = TypeVar("_META1_WITH_SGD", bound=WithBilevelSgdParameter)


class BilevelInterpreter(
    Generic[_META1_WITH_SGD, T, E],
    Meta1ActivationParameter[_META1_WITH_SGD, T, E],
    Meta1Hyperparameter[_META1_WITH_SGD],
):
    def __init__(self, meta0Dialect: _Meta0Dialect[_META1_WITH_SGD, T, E]) -> None:
        Meta1ActivationParameter[_META1_WITH_SGD, T, E].__init__(self, meta0Dialect)


@dataclass(frozen=True, slots=True)
class _Oho_Sgd(WithBilevelSgdParameter, WithOhoPast, Protocol):
    pass


_OHO_SGD = TypeVar("_OHO_SGD", bound=_Oho_Sgd)


class BilevelWithOhoInterpreter(
    Generic[_OHO_SGD, T, E],
    BilevelInterpreter[_OHO_SGD, T, E],
    IsInfluenceTensorOHO[_OHO_SGD],
):
    def __init__(self, meta0Dialect: _Meta0Dialect[_OHO_SGD, T, E]) -> None:
        BilevelInterpreter[_OHO_SGD, T, E].__init__(self, meta0Dialect)


@dataclass(frozen=True, slots=True)
class _Rflo_Sgd(_Oho_Sgd, WithBilevelRflo, Protocol):
    pass


_RFLO_SGD = TypeVar("_RFLO_SGD", bound=_Rflo_Sgd)


class RfloInterpreter(
    Generic[_RFLO_SGD, T, E],
    BilevelWithOhoInterpreter[_RFLO_SGD, T, E],
    IsRflo[_RFLO_SGD],
):
    def __init__(self, meta0Dialect: _Meta0Dialect[_RFLO_SGD, T, E]) -> None:
        BilevelWithOhoInterpreter[_RFLO_SGD, T, E].__init__(self, meta0Dialect)
