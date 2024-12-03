from dataclasses import dataclass
from typing import Callable, Generic, TypeVar
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms



def block_orthogonal_init(size, block_size, frequency=torch.pi/8, amplitude=(0.9, 0.999)):
    W = torch.zeros((size, size))
    num_blocks = size // block_size
    for i in range(num_blocks):
        theta = torch.distributions.uniform.Uniform(-frequency, frequency).sample((1,)) # Random small rotation
        amplitude_val = torch.distributions.uniform.Uniform(*amplitude).sample((1,))     # Random amplitude scaling
        rotation_matrix = amplitude_val * torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                                                        [torch.sin(theta), torch.cos(theta)]])
        
        start = i * block_size
        W[start:start+2, start:start+2] = rotation_matrix  # Assign 2x2 rotation matrix to block
    return W


def initializeParameters(n_in: int, n_h: int, n_out: int
                        ) -> tuple[torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter]:
    
    W_in = torch.nn.init.normal_(torch.empty(n_h, n_in), 0, torch.sqrt(1 / torch.tensor(n_in)))
    W_rec = torch.linalg.qr(torch.normal(0, 1, size=(n_h, n_h)))[0]
    W_out = torch.nn.init.normal_(torch.empty(n_out, n_h), 0, torch.sqrt(1 / torch.tensor(n_h)))
    
    b_rec = torch.zeros(n_h)
    b_out = torch.zeros(n_out)

    # W_in = torch.nn.init.normal_(torch.empty(n_h, n_in), 0, torch.sqrt(1 / torch.tensor(n_in)))
    # W_rec = block_orthogonal_init(n_h, 2)
    # W_out = torch.nn.init.normal_(torch.empty(n_out, n_h), 0, torch.sqrt(1 / torch.tensor(n_h)))

    return W_rec, W_in, b_rec, W_out, b_out


def gru_init(rnn, fc, config):

    W_in = torch.nn.init.normal_(torch.empty(3*config.n_h, config.n_in), 0, torch.sqrt(1 / torch.tensor(3*config.n_in)))
    W_rec = torch.linalg.qr(torch.normal(0, 1, size=(3*config.n_h, config.n_h)))[0]
    W_out = torch.nn.init.normal_(torch.empty(config.n_out, config.n_h), 0, torch.sqrt(1 / torch.tensor(config.n_h)))
    
    b_rec = torch.zeros(3*config.n_h)
    b_out = torch.zeros(config.n_out)

    with torch.no_grad():
        rnn.weight_ih_l0.copy_(W_in)
        rnn.weight_hh_l0.copy_(W_rec)
        rnn.bias_ih_l0.copy_(b_rec)
        rnn.bias_hh_l0.copy_(b_rec)
        fc.weight.copy_(W_out)
        fc.bias.copy_(b_out)

@dataclass
class RnnConfig:
    n_in: int
    n_h: int
    n_out: int
    num_layers: int
    scheme: InitType

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
        
        # gru_init(self.rnn, self.fc, config)

        with torch.no_grad():
            W_rec, W_in, b_rec, W_out, b_out = initializeParameters(config.n_in, config.n_h, config.n_out)
            self.rnn.weight_ih_l0.copy_(W_in)
            self.rnn.weight_hh_l0.copy_(W_rec)
            self.rnn.bias_ih_l0.copy_(b_rec)
            self.rnn.bias_hh_l0.copy_(b_rec)
            self.fc.weight.copy_(W_out)
            self.fc.bias.copy_(b_out)

        # for name, param in self.rnn.named_parameters():
        #     if 'weight' in name:
        #         # torch.nn.init.orthogonal_
        #         torch.nn.init.orthogonal_(param)
        #     elif 'bias' in name:
        #         # torch.nn.init.uniform_(param, a=0.01, b=0.1)
        #         torch.nn.init.zeros_(param)  # It’s often a good practice to zero the biases
        # torch.nn.init.orthogonal_(self.fc.weight)
        # torch.nn.init.zeros_(self.fc.bias)

        self.interface = RNNInterface(
            baseCase=getRNNInit(config.scheme)(config.num_layers, config.n_h),
            forward=lambda x, s0: self.rnn(x, s0)[0]
        )
        # self.gru = nn.GRU(config.n_in, config.n_h, config.num_layers, batch_first=True)
        # self.lstm = nn.LSTM(config.n_in, config.n_h, config.num_layers, batch_first=True)
        # self.interface = RNNInterface(
        #     baseCase=lambda x: (torch.zeros(config.num_layers, x.size(0), config.n_h), torch.zeros(config.num_layers, x.size(0), config.n_h)),
        #     forward=lambda x, s0: self.rnn(x, s0)[0]
        # )
        
    def forward(self, x):
        s0 = self.interface.baseCase(x)
        out = self.interface.forward(x, s0)
        out = self.fc(out)
        return out



@dataclass
class ZeroInit:
    pass

@dataclass
class RandomInit:
    pass

InitType = ZeroInit | RandomInit

def getRNNInit(initScheme: InitType):
    match task:
        case ZeroInit():
            return lambda num_layers, n_h: lambda x: torch.zeros(num_layers, x.size(0), n_h)
        case RandomInit():
            return lambda num_layers, n_h: lambda x: torch.randn(num_layers, x.size(0), n_h)
        case _:
            raise Exception("Invalid dataset type")