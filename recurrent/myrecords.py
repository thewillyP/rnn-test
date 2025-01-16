from typing import Generic, NamedTuple
from torch.utils import _pytree as pytree
from recurrent.mytypes import *


class RnnInferenceState(NamedTuple, Generic[A]):
    activation: ACTIVATION
    parameter: A


class RnnPastFaceState(NamedTuple, Generic[A, B]):
    activation: ACTIVATION
    influenceTensor: INFLUENCETENSOR
    parameter: A
    hyperparameter: B


class BilevelFutureFaceState(NamedTuple, Generic[A, B, C]):
    activation: ACTIVATION
    parameter: A
    hyperparameter: B
    metaHyperparameter: C


class OhoFutureFaceState(NamedTuple, Generic[A, B, C]):
    activation: ACTIVATION
    ohoInfluenceTensor: INFLUENCETENSOR
    parameter: A
    hyperparameter: B
    metaHyperparameter: C


class BilevelPastFaceState(NamedTuple, Generic[A, B, C]):
    activation: ACTIVATION
    influenceTensor: INFLUENCETENSOR
    parameter: A
    hyperparameter: B
    metaHyperparameter: C


class OhoPastFaceState(NamedTuple, Generic[A, B, C]):
    activation: ACTIVATION
    influenceTensor: INFLUENCETENSOR
    ohoInfluenceTensor: INFLUENCETENSOR
    parameter: A
    hyperparameter: B
    metaHyperparameter: C


class RnnFutureFaceState(NamedTuple, Generic[A, B]):
    activation: ACTIVATION
    parameter: A
    hyperparameter: B


T_TORCH = TypeVar("T_TORCH", bound=torch.Tensor | PYTREE)
E_TORCH = TypeVar("E_TORCH", bound=torch.Tensor | PYTREE)


def rnnFutureFaceState_flatten(rnnFutureS: RnnFutureFaceState[T_TORCH, E_TORCH]):
    return (
        rnnFutureS.activation,
        rnnFutureS.parameter,
        rnnFutureS.hyperparameter,
    ), None


def rnnFutureFaceState_unflatten(children: tuple[ACTIVATION, T_TORCH, E_TORCH], aux):
    activation, parameter, hyperparameter = children
    return RnnFutureFaceState[T_TORCH, E_TORCH](
        activation=activation, parameter=parameter, hyperparameter=hyperparameter
    )


pytree.register_pytree_node(
    RnnFutureFaceState,
    rnnFutureFaceState_flatten,
    rnnFutureFaceState_unflatten,
)


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
