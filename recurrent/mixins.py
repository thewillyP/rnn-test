from typing import Protocol, Self
from recurrent.mytypes import *
from recurrent.parameters import RfloConfig


# This is just an entity component model  https://news.ycombinator.com/item?id=7496968
# DO NOT USE NESTED INHERITANCE, SHALLOW MIXINS ONLY
# Even if 2^n combinations, philopsophy is you only need to write code on demand.
# Ideally you could even avoid that with a whole EC system but I'm too lazy to write that.


class WithBasePast(Protocol):
    influenceTensor: INFLUENCETENSOR

    def __replace__(self, **kwargs) -> Self:
        pass


class WithOhoPast(Protocol):
    ohoInfluenceTensor: INFLUENCETENSOR

    def __replace__(self, **kwargs) -> Self:
        pass


class WithParameter(Protocol[T]):
    parameter: T

    def __replace__(self, **kwargs) -> Self:
        pass


class WithHyperparameter(Protocol[T]):
    hyperparameter: T

    def __replace__(self, **kwargs) -> Self:
        pass


class WithBilevelParameter(Protocol[T]):
    metaHyperparameter: T

    def __replace__(self, **kwargs) -> Self:
        pass


class WithRnnActivation(Protocol):
    activation: ACTIVATION

    def __replace__(self, **kwargs) -> Self:
        pass


class WithBaseRflo(Protocol):
    rfloConfig: RfloConfig

    def __replace__(self, **kwargs) -> Self:
        pass


class WithBilevelRflo(Protocol):
    rfloConfig_bilevel: RfloConfig

    def __replace__(self, **kwargs) -> Self:
        pass
