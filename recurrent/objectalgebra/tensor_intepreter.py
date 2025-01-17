from recurrent.datarecords import Input2Output1
from recurrent.objectalgebra.typeclasses import HasInput, HasLabel, HasPredictionInput
from recurrent.mytypes import *
import torch


class IsInput(HasInput[Input2Output1, torch.Tensor]):
    @staticmethod
    def getInput(d: Input2Output1) -> torch.Tensor:
        return d.x


class IsLabel(HasLabel[Input2Output1, torch.Tensor]):
    @staticmethod
    def getLabel(d: Input2Output1) -> torch.Tensor:
        return d.y


class IsPrediction(HasPredictionInput[Input2Output1, torch.Tensor]):
    _prediction_input = torch.empty(0)

    @staticmethod
    def getPredictionInput(_: Input2Output1) -> torch.Tensor:
        return IsPrediction._prediction_input


class Input2Output1Interpreter(IsInput, IsLabel, IsPrediction):
    pass
