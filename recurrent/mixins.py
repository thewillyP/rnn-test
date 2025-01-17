from typing import Protocol, Self
from recurrent.mytypes import *
from recurrent.parameters import RfloConfig


# This is just an entity component model  https://news.ycombinator.com/item?id=7496968
# DO NOT USE NESTED INHERITANCE, SHALLOW MIXINS ONLY
# Even if 2^n combinations, philopsophy is you only need to write code on demand.
# Ideally you could even avoid that with a whole EC system but I'm too lazy to write that.

# See this https://mypy.readthedocs.io/en/stable/common_issues.html#covariant-subtyping-of-mutable-protocol-members-is-rejected
# for why need to have properties as functions. tldlr need explicit deny mutability


class WithBasePast(Protocol):

    @property
    def influenceTensor(self) -> INFLUENCETENSOR:
        pass

    def __replace__(self, **kwargs) -> Self:
        pass


class WithOhoPast(Protocol):
    @property
    def ohoInfluenceTensor(self) -> INFLUENCETENSOR:
        pass

    def __replace__(self, **kwargs) -> Self:
        pass


T_CO = TypeVar("T_CO", covariant=True)


class WithParameter(Protocol[T_CO]):
    @property
    def parameter(self) -> T_CO:
        pass

    def __replace__(self, **kwargs) -> Self:
        pass


class WithHyperparameter(Protocol[T_CO]):
    @property
    def hyperparameter(self) -> T_CO:
        pass

    def __replace__(self, **kwargs) -> Self:
        pass


class WithBilevelParameter(Protocol[T_CO]):
    @property
    def metaHyperparameter(self) -> T_CO:
        pass

    def __replace__(self, **kwargs) -> Self:
        pass


class WithRnnActivation(Protocol):
    @property
    def activation(self) -> ACTIVATION:
        pass

    def __replace__(self, **kwargs) -> Self:
        pass


class WithBaseRflo(Protocol):
    @property
    def rfloConfig(self) -> RfloConfig:
        pass

    def __replace__(self, **kwargs) -> Self:
        pass


class WithBilevelRflo(Protocol):
    @property
    def rfloConfig_bilevel(self) -> RfloConfig:
        pass

    def __replace__(self, **kwargs) -> Self:
        pass
