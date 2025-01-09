from recurrent.datarecords import Input2Output1
from recurrent.objectalgebra.typeclasses import HasInput, HasLabel, HasPredictionInput
from recurrent.mytypes import *
import torch


class IsInput(HasInput[Input2Output1, torch.Tensor]):
    @staticmethod
    def getInput(d: Input2Output1) -> torch.Tensor:
        return torch.cat((d.x1, d.x2), dim=0)


class IsLabel(HasLabel[Input2Output1, torch.Tensor]):
    @staticmethod
    def getLabel(d: Input2Output1) -> torch.Tensor:
        return d.y


class IsPrediction(HasPredictionInput[Input2Output1, torch.Tensor]):
    @staticmethod
    def getPredictionInput(_: Input2Output1) -> torch.Tensor:
        return torch.empty(0)


class Input2Output1Interpreter(
    HasInput[Input2Output1, torch.Tensor],
    HasLabel[Input2Output1, torch.Tensor],
    HasPredictionInput[Input2Output1, torch.Tensor],
):
    pass
