from typing import TypeVar, NewType
import torch

T = TypeVar('T')
E = TypeVar('E')
A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')
D = TypeVar('D')
X = TypeVar('X')
Y = TypeVar('Y')
H = TypeVar('H')
P = TypeVar('P')
L = TypeVar('L')
HP = TypeVar('HP')

ENV = TypeVar('ENV')

DATA = TypeVar('DATA')

# ================== Newtypes =============

ACTIVATION = NewType('ACTIVATION', torch.Tensor)
PARAMETER = NewType('PARAMETER', torch.Tensor)
INFLUENCETENSOR = NewType('INFLUENCETENSOR', torch.Tensor)
HYPERPARAMETER = NewType('HYPERPARAMETER', torch.Tensor)
PREDICTION = NewType('PREDICTION', torch.Tensor)
GRADIENT = NewType('GRADIENT', torch.Tensor)
LOSS = NewType('LOSS', torch.dtype)  # float32
INPUTFEATURE = NewType('INPUTFEATURE', torch.Tensor)
OUTPUTFEATURE = NewType('OUTPUTFEATURE', torch.Tensor)
METAHYPERPARAMETER = NewType('METAHYPERPARAMETER', torch.Tensor)
