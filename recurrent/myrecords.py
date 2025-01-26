from recurrent.mytypes import *
from recurrent.parameters import RfloConfig, RnnConfig, UORO_Param

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
    rfloConfig: RfloConfig = eqx.field(static=True)
    rfloConfig_bilevel: RfloConfig = eqx.field(static=True)
    uoro: UORO_Param[A]
    prng: PRNG


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
        rfloConfig=None,
        rfloConfig_bilevel=None,
        uoro=None,
        prng=None,
    )


def batch_vanilla[
    A: eqx.Module, B: eqx.Module, C: eqx.Module
]() -> RnnGodState[A, B, C]:
    return RnnGodState(
        activation=0,
        influenceTensor=None,
        ohoInfluenceTensor=None,
        parameter=None,
        hyperparameter=None,
        metaHyperparameter=None,
        rnnConfig=None,
        rnnConfig_bilevel=None,
        rfloConfig=None,
        rfloConfig_bilevel=None,
        uoro=None,
        prng=None,
    )


# A_TORCH = TypeVar("A_TORCH", bound=torch.Tensor | PYTREE)
# B_TORCH = TypeVar("B_TORCH", bound=torch.Tensor | PYTREE)
# C_TORCH = TypeVar("C_TORCH", bound=torch.Tensor | PYTREE)


# def rnnGodState_flatten(godState: RnnGodState[A_TORCH, B_TORCH, C_TORCH]):
#     return (
#         godState.activation,
#         godState.influenceTensor,
#         godState.ohoInfluenceTensor,
#         godState.parameter,
#         godState.hyperparameter,
#         godState.metaHyperparameter,
#     ), (godState.rfloConfig, godState.rfloConfig_bilevel)


# def rnnGodState_unflatten(
#     children: tuple[
#         ACTIVATION, INFLUENCETENSOR, INFLUENCETENSOR, A_TORCH, B_TORCH, C_TORCH
#     ],
#     aux: tuple[RfloConfig, RfloConfig],
# ):
#     (
#         activation,
#         influenceTensor,
#         ohoInfluenceTensor,
#         parameter,
#         hyperparameter,
#         metaHyperparameter,
#     ) = children
#     (rfloConfig, rfloConfig_bilevel) = aux
#     return RnnGodState(
#         activation=activation,
#         influenceTensor=influenceTensor,
#         ohoInfluenceTensor=ohoInfluenceTensor,
#         parameter=parameter,
#         hyperparameter=hyperparameter,
#         metaHyperparameter=metaHyperparameter,
#         rfloConfig=rfloConfig,
#         rfloConfig_bilevel=rfloConfig_bilevel,
#     )


# pytree.register_pytree_node(
#     RnnGodState,
#     rnnGodState_flatten,
#     rnnGodState_unflatten,
# )


# @dataclass(frozen=True)
# class ArtifactConfig:
#     artifact: Callable[[str], wandb.Artifact]
#     path: Callable[[str], str]


# class Logger(ABC):

#     @abstractmethod
#     def log2External(self, artifactConfig: ArtifactConfig, saveFile: Callable[[str], None], name: str):
#         pass

#     @abstractmethod
#     def log(self, dict: dict[str, Any]):
#         pass

#     @abstractmethod
#     def init(self, projectName: str, config: argparse.Namespace):
#         pass

#     @abstractmethod
#     def watchPytorch(self, model: torch.nn.Module):
#         pass


# @dataclass(frozen=True)
# class ZeroInit:
#     pass

# @dataclass(frozen=True)
# class RandomInit:
#     pass

# @dataclass(frozen=True)
# class StaticRandomInit:
#     pass

# InitType = ZeroInit | RandomInit | StaticRandomInit


# # @dataclass(frozen=True)
# # class RELU:
# #     pass

# # @dataclass(frozen=True)
# # class TANH:
# #     pass

# # ActivationFnType = RELU | TANH


# @dataclass(frozen=True)
# class RnnConfig:
#     n_in: int
#     n_h: int
#     n_out: int
#     num_layers: int
#     scheme: InitType
#     activation: Callable[[torch.Tensor], torch.Tensor]


# @dataclass(frozen=True)
# class Uniform:
#     pass

# @dataclass(frozen=True)
# class Normal:
#     pass

# RandomType = Uniform | Normal

# @dataclass(frozen=True)
# class Random:
#     random_type: RandomType

# @dataclass(frozen=True)
# class Sparse:
#     outT: float

# @dataclass(frozen=True)
# class Wave:
#     pass

# DatasetType = Random | Sparse | Wave

# # TODO make it into a separate config, make adt Type = Oho | NoOho you get the drill
# # @dataclass(frozen=True)
# # class OhoConfig:


# @dataclass(frozen=True)
# class Config:
#     task: DatasetType
#     seq: int
#     numTr: int
#     numVl: int
#     numTe: int
#     batch_size_tr: int
#     batch_size_vl: int
#     batch_size_te: int
#     t1: float
#     t2: float
#     num_epochs: int
#     learning_rate: float
#     l2_regularization: float
#     rnnConfig: RnnConfig
#     criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
#     optimizerFn: Callable
#     modelArtifact: ArtifactConfig
#     datasetArtifact: ArtifactConfig
#     checkpointFrequency: int
#     projectName: str
#     seed: int
#     performanceSamples: int
#     logFrequency: int
#     meta_learning_rate: float
#     is_oho: bool
#     time_chunk_size: int
#     rnnInitialActivation: Callable[[int], torch.Tensor]
