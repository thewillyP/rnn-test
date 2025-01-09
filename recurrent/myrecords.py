from dataclasses import dataclass
from typing import Callable, Protocol
import torch
from recurrent.mytypes import *
from torch.utils import _pytree as pytree


# This is just an entity component model  https://news.ycombinator.com/item?id=7496968
# DO NOT USE NESTED INHERITANCE, SHALLOW MIXINS ONLY
# Even if 2^n combinations, philopsophy is you only need to write code on demand.
# Ideally you could even avoid that with a whole EC system but I'm too lazy to write that.


@dataclass(frozen=True)
class WithRnnConfig(Protocol):
    n_h: int
    n_in: int
    n_out: int
    alpha: float
    activationFn: Callable[[torch.Tensor], torch.Tensor]


# @dataclass(frozen=True)
# class WithTrainPrediction(Protocol):
#     trainPrediction: PREDICTION


# @dataclass(frozen=True)
# class WithTrainLoss(Protocol):
#     trainLoss: LOSS


# @dataclass(frozen=True)
# class WithTrainGradient(Protocol):
#     trainGradient: GRADIENT


# @dataclass(frozen=True)
# class WithValidationPrediction(Protocol):
#     validationPrediction: PREDICTION


# @dataclass(frozen=True)
# class WithValidationLoss(Protocol):
#     validationLoss: LOSS


# @dataclass(frozen=True)
# class WithValidationGradient(Protocol):
#     validationGradient: GRADIENT


@dataclass(frozen=True)
class WithHyperparameter(Protocol):
    hyperparameter: HYPERPARAMETER


@dataclass(frozen=True)
class WithBaseFuture(Protocol):
    influenceTensor: INFLUENCETENSOR


@dataclass(frozen=True)
class WithOhoFuture(Protocol):
    ohoInfluenceTensor: INFLUENCETENSOR


@dataclass(frozen=True)
class WithBilevel(Protocol):
    metaHyperparameter: METAHYPERPARAMETER


@dataclass(frozen=True)
class WithRnn(Protocol):
    activation: ACTIVATION
    parameter: PARAMETER


@dataclass(frozen=True)
class RfloConfig:
    rflo_alpha: float


@dataclass(frozen=True)
class WithBaseRflo(Protocol):
    rfloConfig: RfloConfig


@dataclass(frozen=True)
class WithBilevelRflo(Protocol):
    rfloConfig_bilevel: RfloConfig


@dataclass(frozen=True)
class RnnEnv(WithRnnConfig, WithRnn):
    pass


@dataclass(frozen=True)
class RnnLearnable(RnnEnv, WithHyperparameter):
    pass


@dataclass(frozen=True)
class RnnBPTTState(RnnLearnable):
    pass


@dataclass(frozen=True)
class RnnFutureState(RnnLearnable, WithBaseFuture):
    pass


@dataclass(frozen=True)
class BilevelLearnable(
    RnnLearnable,
    WithBilevel,
):
    pass


@dataclass(frozen=True)
class BilevelBPTTState(BilevelLearnable):
    pass


@dataclass(frozen=True)
class OhoBPTTState(BilevelLearnable, WithOhoFuture):
    pass


@dataclass(frozen=True)
class BilevelFutureState(BilevelLearnable, WithBaseFuture):
    pass


@dataclass(frozen=True)
class OhoFutureState(BilevelLearnable, WithOhoFuture, WithBaseFuture):
    pass


def rnnBPTTState_flatten(rnnBpttState: RnnBPTTState):
    return (rnnBpttState.activation,), (
        rnnBpttState.parameter,
        rnnBpttState.hyperparameter,
        rnnBpttState.n_h,
        rnnBpttState.n_in,
        rnnBpttState.n_out,
        rnnBpttState.alpha,
        rnnBpttState.activationFn,
    )


def rnnBPTTState_unflatten(children, aux):
    a = children[0]
    p, hp, n_h, n_in, n_out, alpha, activationFn = aux
    return RnnBPTTState(
        activation=a,
        parameter=p,
        hyperparameter=hp,
        n_h=n_h,
        n_in=n_in,
        n_out=n_out,
        alpha=alpha,
        activationFn=activationFn,
    )


pytree.register_pytree_node(RnnBPTTState, rnnBPTTState_flatten, rnnBPTTState_unflatten)


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
