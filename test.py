#%%
from itertools import cycle
import time
import torch 
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from delayed_add_task import getDataLoaderIO, DatasetType, randomUniform, sparseIO, waveIO, waveArbitraryUniform, sparseUniformConstOutT, visualizeOutput
from learning import *
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import wandb
from matplotlib.ticker import MaxNLocator
import torch.nn as nn
from torchvision import datasets, transforms
from abc import ABCMeta, abstractmethod
from typing import TypeVar, Generic
import torch.nn.functional as F
from dataclasses import dataclass
from torchviz import make_dot


T = TypeVar('T')
E = TypeVar('E')
A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')
D = TypeVar('D')
X = TypeVar('X')
Y = TypeVar('Y')


def jacobian(outs, inps) -> torch.Tensor:
    outs = torch.atleast_1d(outs)
    I_N = torch.eye(outs.size(0))
    def get_vjp(v):
        return torch.autograd.grad(outs, inps, v, create_graph=True, retain_graph=True)[0]
    return torch.vmap(get_vjp)(I_N)


@dataclass(frozen=True)
class ForwardFacingLearning(metaclass=ABCMeta):
    getInfluenceTensor: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    getRecurrentGradient: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def createForwardFacingLearning(getInfluenceTensor: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]) -> ForwardFacingLearning:
    def getRecurrentGradient(
                    influenceTensor: torch.Tensor
                    , prevDynamic: torch.Tensor
                    , dynamic: torch.Tensor
                    , loss: torch.Tensor
                    , parameter: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        immediateCreditAssignment = jacobian(loss, dynamic)
        influenceTensor_ = getInfluenceTensor(influenceTensor, prevDynamic, dynamic, parameter)
        recGrad_ = immediateCreditAssignment @ influenceTensor_
        return recGrad_, influenceTensor_
    return ForwardFacingLearning(getInfluenceTensor, getRecurrentGradient)

def RTRL() -> ForwardFacingLearning:
    def getInfluenceTensor(influenceTensor: torch.Tensor
                        , prevDynamic: torch.Tensor
                        , dynamic: torch.Tensor
                        , parameter: torch.Tensor) -> torch.Tensor:
        immediateJacobian = jacobian(dynamic, prevDynamic)
        immediateInfluence = jacobian(dynamic, parameter)
        return immediateJacobian @ influenceTensor + immediateInfluence
    return createForwardFacingLearning(getInfluenceTensor)


def RFLO(alpha: float) -> ForwardFacingLearning:
    def getInfluenceTensor(influenceTensor: torch.Tensor
                        , _: torch.Tensor
                        , dynamic: torch.Tensor
                        , parameter: torch.Tensor) -> torch.Tensor:
        immediateInfluence = jacobian(dynamic, parameter)
        return (1 - alpha) * influenceTensor + alpha * immediateInfluence
    return createForwardFacingLearning(getInfluenceTensor)


torch.manual_seed(0)

rtrl = RTRL()

n = 5 
n_in = 2 
n_out = 1

x_ = torch.randn(n_in)
prevDynamic = torch.zeros(n, requires_grad=True)
influenceTensor = torch.zeros(n, n*(n+n_in+1), requires_grad=True)

parameters = torch.randn(n*(n+n_in+1) + n_out*(n+1), requires_grad=True)

W_rec, W_out = torch.split(parameters, [n*(n+n_in+1), n_out*(n+1)])
# W_rec = torch.randn(n*(n+n_in+1), requires_grad=True)
# W_out = torch.randn(n_out*(n+1), requires_grad=True)

def network(x, a, w_rec, alpha):
    return (1 - alpha) * a + alpha * torch.relu(torch.reshape(w_rec, (n, n+n_in+1)) @ torch.cat((x, a, torch.tensor([1.0]))))

def readout(a, w_out):
    return torch.reshape(w_out, (n_out, n+1)) @ torch.cat((a, torch.tensor([1.0])))

dynamic = network(x_, prevDynamic, W_rec, 1)
output = readout(dynamic, W_out)
loss = F.mse_loss(output, torch.tensor([1.0])) 

immediateCreditAssignment = jacobian(loss, dynamic)
immediateJacobian = jacobian(dynamic, prevDynamic)
immediateInfluence = jacobian(dynamic, W_rec)
influenceTensor_ = immediateJacobian @ influenceTensor + immediateInfluence
recGrad = immediateCreditAssignment @ influenceTensor_
readoutGrad = jacobian(loss, W_out)


grad = torch.squeeze(torch.cat((recGrad, readoutGrad), dim=1))
parameters_ = parameters - 0.01 * grad

parameters_ = parameters_.detach().requires_grad_()
influenceTensor_ = influenceTensor_.detach().requires_grad_()
dynamic = dynamic.detach().requires_grad_()

W_rec_, W_out_ = torch.split(parameters_, [n*(n+n_in+1), n_out*(n+1)])

dynamic_ = network(x_, dynamic, W_rec_, 1)
output_ = readout(dynamic_, W_out_)
loss_ = F.mse_loss(output_, torch.tensor([1.0]))

immediateCreditAssignment_ = jacobian(loss_, dynamic_)
immediateJacobian_ = jacobian(dynamic_, dynamic)
immediateInfluence_ = jacobian(dynamic_, W_rec_)
influenceTensor__ = immediateJacobian_ @ influenceTensor_ + immediateInfluence_
recGrad_ = immediateCreditAssignment_ @ influenceTensor__
readoutGrad_ = jacobian(loss_, W_out_)

grad_ = torch.squeeze(torch.cat((recGrad_, readoutGrad_), dim=1))
parameters__ = parameters_ - 0.01 * grad_



rtrlParameters = parameters.clone().requires_grad_(True)
rtrlPrevDynamic = prevDynamic.clone().requires_grad_(True)
rtrlIM = influenceTensor.clone().requires_grad_(True)
for _ in range(2):
    rtrl_W_rec, rtrl_W_out = torch.split(rtrlParameters, [n*(n+n_in+1), n_out*(n+1)])

    rtrlDynamic = network(x_, rtrlPrevDynamic, rtrl_W_rec, 1)
    rtrlOutput = readout(rtrlDynamic, rtrl_W_out)
    rtrlLoss = F.mse_loss(rtrlOutput, torch.tensor([1.0])) 

    rtrl_W_rec_Grad, rtrlIM = rtrl.getRecurrentGradient(rtrlIM, rtrlPrevDynamic, rtrlDynamic, rtrlLoss, rtrl_W_rec)
    rtrl_W_out_Grad = jacobian(rtrlLoss, rtrl_W_out)

    # Combine gradients and update parameters
    rtrl_grad = torch.squeeze(torch.cat((rtrl_W_rec_Grad, rtrl_W_out_Grad), dim=1))
    # rtrlParameters = rtrlParameters - 0.01 * rtrl_grad

    # Detach and require gradients for the next step
    # rtrlParameters = rtrlParameters.detach().requires_grad_()
    rtrlPrevDynamic = rtrlDynamic.detach().requires_grad_()
    rtrlIM = rtrlIM.detach().requires_grad_()



# assert torch.allclose(parameters__, rtrlParameters)

W_rec = torch.cat((torch.eye(2), torch.eye(2), torch.zeros(2, 1)), dim=1).requires_grad_(True)
W_out = torch.cat((torch.eye(2), torch.zeros(2, 1)), dim=1).requires_grad_(True)
print(W_rec)
print(W_out)
print(W_rec.shape)
quit()

# class Test_RTRL():

#     @classmethod
#     def setUpClass(cls):


#         W_rec = torch.cat((torch.eye(2), torch.eye(2), torch.zeros(2)), dim=1)
#         W_out = torch.cat((torch.eye(2), torch.zeros(2)), dim=1)

#         cls.W_in = np.eye(2)
#         cls.W_rec = np.eye(2)
#         cls.W_out = np.eye(2)
#         cls.b_rec = np.zeros(2)
#         cls.b_out = np.zeros(2)

#         cls.rnn = RNN(cls.W_in, cls.W_rec, cls.W_out,
#                       cls.b_rec, cls.b_out,
#                       activation=identity,
#                       alpha=1,
#                       output=softmax,
#                       loss=softmax_cross_entropy)

#         cls.rnn.a = np.ones(2)
#         cls.rnn.a_prev = np.ones(2)
#         cls.rnn.x = np.ones(2) * 2
#         cls.rnn.error = np.ones(2) * 0.5
#         cls.rnn.a_J = np.eye(2)

#     def test_update_learning_vars(self):

#         self.learn_alg = RTRL(self.rnn)
#         self.learn_alg.dadw += 1
#         self.learn_alg.update_learning_vars()
#         papw = np.concatenate([np.eye(2), np.eye(2),
#                                np.eye(2) * 2, np.eye(2) * 2, np.eye(2)], axis=1)
#         correct_dadw = np.ones((2, 10)) + papw
#         assert_allclose(self.learn_alg.dadw, correct_dadw)

#     def test_get_rec_grads(self):

#         self.learn_alg = RTRL(self.rnn)
#         self.learn_alg.q = np.ones(2)
#         self.learn_alg.dadw = np.concatenate([np.eye(2) * i for i in range(5)],
#                                              axis=1)
#         rec_grads = self.learn_alg.get_rec_grads()
#         correct_rec_grads = np.array([list(range(5)) for _ in [0, 1]])
#         assert_allclose(rec_grads, correct_rec_grads)



# grad_ = torch.squeeze(torch.cat((recGrad_, readoutGrad_), dim=1))
# parameters__ = parameters_ - 0.01 * grad_

# parameters__ = parameters__.detach().requires_grad_()
# influenceTensor__ = influenceTensor__.detach().requires_grad_()
# dynamic_ = dynamic_.detach().requires_grad_()


# W_rec__, W_out__ = torch.split(parameters__, [n*(n+n_in+1), n_out*(n+1)])

# dynamic__ = network(x_, dynamic_, W_rec__, 1)
# output__ = readout(dynamic__, W_out__)
# loss__ = F.mse_loss(output__, torch.tensor([1.0]))

# immediateCreditAssignment__ = jacobian(loss__, dynamic__)
# immediateJacobian__ = jacobian(dynamic__, dynamic_)
# immediateInfluence__ = jacobian(dynamic__, W_rec__)
# influenceTensor___ = immediateJacobian__ @ influenceTensor__ + immediateInfluence__
# recGrad__ = immediateCreditAssignment__ @ influenceTensor__
# readoutGrad__ = jacobian(loss__, W_out__)


# make_dot(loss__, params={"AAAAAAAAAAAAAAAAA": parameters__, "prevDynamic": dynamic_, "test": dynamic, "asdf": parameters_}).render("testering10", format="png")



# do a test to see if im wastefully building comp graph

# print(immediateCreditAssignment)
# print(immediateJacobian)
# print(immediateJacobian.shape)
# print(immediateInfluence)
# print(immediateInfluence.shape)
# print(influenceTensor_)
# print(influenceTensor_.shape)
# print(recGrad)
# print(recGrad.shape)
# print(readoutGrad)

# print(immediateCreditAssignment_)
# print(immediateJacobian_)
# print(immediateInfluence_)
# print(influenceTensor__)
# print(recGrad_)


testy = lambda pa, w: network(x_, pa, w, 1)
tester = torch.autograd.grad(loss, W_rec, retain_graph=True)[0]
tett = torch.autograd.grad(loss, W_out, retain_graph=True)[0]
assert torch.allclose(immediateJacobian, torch.func.jacrev(testy, argnums=0)(prevDynamic, W_rec))
assert torch.allclose(immediateInfluence, torch.func.jacrev(testy, argnums=1)(prevDynamic, W_rec))
assert torch.allclose(recGrad, tester)
assert torch.allclose(readoutGrad, tett)



