from dataclasses import dataclass
from recurrent.mytypes import *
from recurrent.parameters import Logs, RnnConfig, UORO_Param
from jax import Array

"""
Expression problem is very hard to solve in python, without there being some resistance.
First, I want 'inheritance' since that's what makes adding new datatypes easy. 
But NamedTuples can't use inheritance and that's bad because
1. NT are automatically PyTrees, I don't want to have to write boilerplate to register if using dataclass
2. NT more memory efficient

Compromise: Everytime I add a field, this file is the only place I should need to change. 

I will have to specify vmap batch dimensions for all fields regardless of whether I solve the EP.

On the good side, Object Algebras still buys me extensible interfaces which is already cracked on its own.

^
|
equinox solves this problem for me with eqx.Module which supports inheritance and is also a PyTree. 
Will refactor later, but global god state is fine for now. 
"""


class RnnGodState[A: eqx.Module, B: eqx.Module, C: eqx.Module](eqx.Module):
    activation: ACTIVATION
    influenceTensor: Gradient[A]
    ohoInfluenceTensor: Gradient[B]
    parameter: A
    hyperparameter: B
    metaHyperparameter: C
    rnnConfig: RnnConfig = eqx.field(static=True)
    rnnConfig_bilevel: RnnConfig = eqx.field(static=True)
    uoro: UORO_Param
    prng: PRNG
    logs: Logs
    oho_logs: Logs


def batch_rtrl[A: eqx.Module, B: eqx.Module, C: eqx.Module]() -> RnnGodState[A, B, C]:
    return RnnGodState(
        activation=0,
        influenceTensor=0,
        ohoInfluenceTensor=None,
        parameter=None,
        hyperparameter=None,
        metaHyperparameter=None,
        rnnConfig=None,
        rnnConfig_bilevel=None,
        uoro=None,
        prng=None,
    )


def batch_vanilla[A: eqx.Module, B: eqx.Module, C: eqx.Module]() -> RnnGodState[A, B, C]:
    return RnnGodState(
        activation=0,
        influenceTensor=None,
        ohoInfluenceTensor=None,
        parameter=None,
        hyperparameter=None,
        metaHyperparameter=None,
        rnnConfig=None,
        rnnConfig_bilevel=None,
        uoro=None,
        prng=None,
    )


# @dataclass(frozen=True)
# class Config:
#     numTr: int
#     numVl: int
#     numTe: int
#     batch_size_tr: int
#     batch_size_vl: int
#     num_epochs: int
#     learning_rate: float
#     l2_regularization: float
#     criterion: Callable[[Array, Array], Array]
#     optimizerFn: Callable
#     checkpointFrequency: int
#     projectName: str
#     seed: int
#     performanceSamples: int
#     logFrequency: int
#     meta_learning_rate: float
#     is_oho: bool
#     time_chunk_size: int
