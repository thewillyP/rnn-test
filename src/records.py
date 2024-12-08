

from abc import ABC, abstractmethod
import argparse
from dataclasses import dataclass
from typing import Any, Callable
import torch
import wandb


@dataclass(frozen=True)
class ArtifactConfig:
    artifact: Callable[[str], wandb.Artifact]
    path: Callable[[str], str]



class Logger(ABC):

    @abstractmethod
    def log2External(self, artifactConfig: ArtifactConfig, saveFile: Callable[[str], None], name: str):
        pass

    @abstractmethod
    def log(self, dict: dict[str, Any]):
        pass

    @abstractmethod
    def init(self, projectName: str, config: argparse.Namespace):
        pass

    @abstractmethod
    def watchPytorch(self, model: torch.nn.Module):
        pass


@dataclass(frozen=True)
class ZeroInit:
    pass

@dataclass(frozen=True)
class RandomInit:
    pass

InitType = ZeroInit | RandomInit


# @dataclass(frozen=True)
# class RELU:
#     pass

# @dataclass(frozen=True)
# class TANH:
#     pass

# ActivationFnType = RELU | TANH


@dataclass(frozen=True)
class RnnConfig:
    n_in: int
    n_h: int
    n_out: int
    num_layers: int
    scheme: InitType
    activation: Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class EfficientBPTT:
    truncation: int

RNNLearningType = EfficientBPTT


@dataclass(frozen=True)
class Random:
    pass

@dataclass(frozen=True)
class Sparse:
    outT: float

@dataclass(frozen=True)
class Wave:
    pass

DatasetType = Random | Sparse | Wave


@dataclass(frozen=True)
class Config:
    task: DatasetType
    seq: int
    numTr: int
    numVl: int
    numTe: int
    batch_size_tr: int
    batch_size_vl: int
    batch_size_te: int
    t1: float
    t2: float
    num_epochs: int
    learning_rate: float
    rnnConfig: RnnConfig
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    optimizerFn: Callable
    modelArtifact: ArtifactConfig  
    datasetArtifact: ArtifactConfig
    checkpointFrequency: int
    projectName: str
    seed: int
    performanceSamples: int
    logFrequency: int
    rnnLearningAlgorithm: RNNLearningType

