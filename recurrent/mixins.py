from dataclasses import dataclass
from typing import Protocol
from recurrent.mytypes import *
from recurrent.parameters import RfloConfig, RnnParameter, SgdParameter


# This is just an entity component model  https://news.ycombinator.com/item?id=7496968
# DO NOT USE NESTED INHERITANCE, SHALLOW MIXINS ONLY
# Even if 2^n combinations, philopsophy is you only need to write code on demand.
# Ideally you could even avoid that with a whole EC system but I'm too lazy to write that.


@dataclass(frozen=True)
class WithBasePast(Protocol):
    influenceTensor: INFLUENCETENSOR


@dataclass(frozen=True)
class WithOhoPast(Protocol):
    ohoInfluenceTensor: INFLUENCETENSOR


@dataclass(frozen=True, slots=True)
class WithRnnParameter(Protocol):
    parameter: RnnParameter


@dataclass(frozen=True, slots=True)
class WithSgdParameter(Protocol):
    hyperparameter: SgdParameter


@dataclass(frozen=True, slots=True)
class WithBilevelSgdParameter(Protocol):
    metaHyperparameter: SgdParameter


@dataclass(frozen=True)
class WithRnnActivation(Protocol):
    activation: ACTIVATION


@dataclass(frozen=True)
class WithBaseRflo(Protocol):
    rfloConfig: RfloConfig


@dataclass(frozen=True)
class WithBilevelRflo(Protocol):
    rfloConfig_bilevel: RfloConfig
