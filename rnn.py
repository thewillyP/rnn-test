from dataclasses import dataclass
from typing import Callable, Generic, TypeVar
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


@dataclass
class RnnConfig:
    n_in: int
    n_h: int
    n_out: int
    num_layers: int

T = TypeVar('T')
X = TypeVar('X')

@dataclass
class RNNInterface(Generic[X, T]):
    baseCase: Callable[[X], T]
    forward: Callable[[X, T], torch.Tensor]

# Fully connected neural network with one hidden layer
class RNN(nn.Module):
    def __init__(self, config: RnnConfig):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(config.n_in, config.n_h, config.num_layers, batch_first=True)
        self.fc = nn.Linear(config.n_h, config.n_out)

        self.interface = RNNInterface(
            baseCase=lambda x: torch.zeros(config.num_layers, x.size(0), config.n_h),
            forward=lambda x, s0: self.rnn(x, s0)[0]
        )
        # self.gru = nn.GRU(config.n_in, config.n_h, config.num_layers, batch_first=True)
        # self.lstm = nn.LSTM(config.n_in, config.n_h, config.num_layers, batch_first=True)
        # self.interface = RNNInterface(
        #     baseCase=lambda x: (torch.zeros(config.num_layers, x.size(0), config.n_h), torch.zeros(self.num_layers, x.size(0), self.hidden_size)),
        #     forward=lambda x, s0: self.lstm(x, s0)[0]
        # )
        
    def forward(self, x):
        s0 = self.interface.baseCase(x)
        out = self.interface.forward(x, s0)
        out = self.fc(out)
        return out


