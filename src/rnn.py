from dataclasses import dataclass
from typing import Callable, Generic, TypeVar
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from records import InitType, RnnConfig, ZeroInit, RandomInit


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

        self.interface = RNNInterface(
            baseCase=getRNNInit(config.scheme)(config.num_layers, config.n_h),
            forward=lambda x, s0: self.rnn(x, s0)[0]
        )


        

        self.param_sizes = [p.numel() for p in self.parameters()]
        self.n_params = sum(self.param_sizes)
        self.param_shapes = [tuple(p.shape) for p in self.parameters()]
        self.param_cumsum = torch.cumsum([0] + self.param_sizes)

        
    def forward(self, x):
        s0 = self.interface.baseCase(x)
        out = self.interface.forward(x, s0)
        out = self.fc(out)
        return out


class MLP(nn.Module):

    def __init__(self, n_layers, layer_sizes, lr_init, lambda_l2, is_cuda=0):
        super(MLP, self).__init__()


        self.layer_sizes = layer_sizes
        self.n_layers = n_layers
        self.n_params = 0
        for i in range(1, self.n_layers):
            attr = 'layer_{}'.format(i)
            layer = nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            if is_cuda: layer = layer.cuda()
            setattr(self, attr, layer)

            param_size = (layer_sizes[i - 1] + 1) * layer_sizes[i]
            self.n_params += param_size

        self.param_sizes = [p.numel() for p in self.parameters()]
        self.param_shapes = [tuple(p.shape) for p in self.parameters()]
        self.param_cumsum = np.cumsum([0] + self.param_sizes)

        self.reset_jacob(is_cuda)
        self.eta  = lr_init
        self.lambda_l2 = lambda_l2
        self.name = 'MLP'
        self.grad_norm = 0
        self.grad_norm_vl = 0
        self.grad_angle = 0
        self.param_norm = 0
        self.dFdlr_norm = 0
        self.dFdl2_nrom = 0

    def reset_jacob(self, is_cuda=1):
        self.dFdlr = torch.zeros(self.n_params) #np.zeros(self.n_params)
        self.dFdl2 = torch.zeros(self.n_params) #np.zeros(self.n_params)
        self.dFdl2_norm = 0
        self.dFdlr_norm = 0
        if is_cuda: 
            self.dFdlr = self.dFdlr.cuda()
            self.dFdl2 = self.dFdl2.cuda()

    def forward(self, x, logsoftmaxF=1):

        x = x.view(-1, self.layer_sizes[0])
        for i_layer in range(1, self.n_layers):
            attr = 'layer_{}'.format(i_layer)
            layer = getattr(self, attr)
            x = layer(x)
            if i_layer < self.n_layers - 1:
                #x = F.relu(x)
                x = torch.tanh(x)
        if logsoftmaxF:
            return F.log_softmax(x, dim=1)
        else:
            return F.softmax(x, dim=1)

    def update_dFdlr(self, Hv, param, grad, is_cuda=0, opt_type='sgd', noise=None, N=50000):

        #grad = flatten_array([p.grad.data.numpy() for p in self.parameters()])
        #tmp = np.ones(self.n_params) * 0.01 
        self.Hlr = self.eta*Hv
        self.Hlr_norm = norm(self.Hlr)
        self.dFdlr_norm = norm(self.dFdlr)
        self.dFdlr.data = self.dFdlr.data * (1-2*self.lambda_l2*self.eta) \
                                - self.Hlr - grad - 2*self.lambda_l2*param
        if opt_type == 'sgld':
            if noise is None: noise = torch.randn(size=param.shape)
            self.dFdlr.data = self.dFdlr.data + 0.5 * torch.sqrt(2*noise / self.eta / N)

    def update_dFdlambda_l2(self, Hv, param, grad, is_cuda=0):
       
        self.Hl2 = self.eta*Hv
        self.Hl2_norm = norm(self.Hl2)
        self.dFdl2_norm = norm(self.dFdl2)
        self.dFdl2.data = self.dFdl2.data * (1-2*self.lambda_l2*self.eta) \
                                                - self.Hl2  - 2*self.eta*param


    def update_eta(self, mlr, val_grad):

        val_grad = flatten_array(val_grad)
        delta = val_grad.dot(self.dFdlr).data.cpu().numpy()
        self.eta -= mlr * delta
        self.eta = np.maximum(0.0, self.eta)

    def update_lambda(self, mlr, val_grad):

        val_grad = flatten_array(val_grad)
        delta = val_grad.dot(self.dFdl2).data.cpu().numpy()
        self.lambda_l2 -= mlr * delta 
        self.lambda_l2 = np.maximum(0, self.lambda_l2)
        self.lambda_l2 = np.minimum(0.0002, self.lambda_l2)



def getRNNInit(initScheme: InitType):
    match initScheme:
        case ZeroInit():
            return lambda num_layers, n_h: lambda x: torch.zeros(num_layers, x.size(0), n_h)
        case RandomInit():
            return lambda num_layers, n_h: lambda x: torch.randn(num_layers, x.size(0), n_h)
        case _:
            raise Exception("Invalid init scheme type")