from typing import Callable
import torch
from dataclasses import dataclass
from mytypes import *


def jacobian(outs, inps) -> torch.Tensor:
    outs = torch.atleast_1d(outs)
    I_N = torch.eye(outs.size(0))

    def get_vjp(v):
        return torch.autograd.grad(outs, inps, v, create_graph=True, retain_graph=True)[
            0
        ]

    return torch.vmap(get_vjp)(I_N)


@dataclass(frozen=True)
class ForwardFacingLearning:
    getInfluenceTensor: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ]
    getRecurrentGradient: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]


def createForwardFacingLearning(
    getInfluenceTensor: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ]
) -> ForwardFacingLearning:
    def getRecurrentGradient(
        influenceTensor: torch.Tensor,
        prevDynamic: torch.Tensor,
        dynamic: torch.Tensor,
        loss: torch.Tensor,
        parameter: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        immediateCreditAssignment = jacobian(loss, dynamic)
        influenceTensor_ = getInfluenceTensor(
            influenceTensor, prevDynamic, dynamic, parameter
        )
        recGrad_ = immediateCreditAssignment @ influenceTensor_
        return recGrad_, influenceTensor_

    return ForwardFacingLearning(getInfluenceTensor, getRecurrentGradient)


def RTRL() -> ForwardFacingLearning:
    def getInfluenceTensor(
        influenceTensor: torch.Tensor,
        prevDynamic: torch.Tensor,
        dynamic: torch.Tensor,
        parameter: torch.Tensor,
    ) -> torch.Tensor:
        immediateJacobian = jacobian(dynamic, prevDynamic)
        immediateInfluence = jacobian(dynamic, parameter)
        return immediateJacobian @ influenceTensor + immediateInfluence

    return createForwardFacingLearning(getInfluenceTensor)


def RFLO(alpha: float) -> ForwardFacingLearning:
    def getInfluenceTensor(
        influenceTensor: torch.Tensor,
        _: torch.Tensor,
        dynamic: torch.Tensor,
        parameter: torch.Tensor,
    ) -> torch.Tensor:
        immediateInfluence = jacobian(dynamic, parameter)
        return (1 - alpha) * influenceTensor + alpha * immediateInfluence

    return createForwardFacingLearning(getInfluenceTensor)
