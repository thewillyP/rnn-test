from dataclasses import dataclass
from typing import Callable, Generic, Iterable, TypeVar
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from records import InitType, RnnConfig, ZeroInit, RandomInit, StaticRandomInit
import wandb

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


# def rnnStep(  config: RnnConfig
#             , x: torch.Tensor
#             , a: torch.Tensor
#             , w_rec: torch.Tensor) -> torch.Tensor:
#     w_rec = torch.reshape(w_rec, (config.n_h, config.n_h+config.n_in+1))
#     return (1 - config.alpha) * a + config.alpha * config.activation(w_rec @ torch.cat((x, a, torch.tensor([1.0]))))

# def rnnReadout(   config: RnnConfig
#                 , a: torch.Tensor
#                 , w_out: torch.Tensor) -> torch.Tensor:
#     w_out = torch.reshape(w_out, (config.n_out, config.n_h+1))
#     return w_out @ torch.cat((a, torch.tensor([1.0])))


# def rnnSplitParameters(parameters: torch.Tensor, config: RnnConfig) -> tuple[torch.Tensor, torch.Tensor]:
#     w_rec, w_out = torch.split(parameters, [config.n_h*(config.n_h+config.n_in+1), config.n_out*(config.n_h+1)])
#     return w_rec, w_out

# def rnnForward( config: RnnConfig
#             , parameters: torch.Tensor
#             , x: Iterable[torch.Tensor]
#             , a: torch.Tensor) -> torch.Tensor:
#     w_rec, w_out = rnnSplitParameters(parameters, config)
#     activations = []
#     for x_ in x:
#         a = rnnStep(config, x_, a, w_rec)
#         activations.append(a)
#     return rnnReadout(config, a, w_out), activations

# def getParametersIO(config: RnnConfig):
#     W_rec, W_in, b_rec, W_out, b_out = initializeParameters(config.n_in, config.n_h, config.n_out)
#     return torch.cat((W_rec.flatten(), W_in.flatten(), b_rec.flatten(), W_out.flatten(), b_out.flatten()))
    

# def update_dFdlr(dFdlr, lr, l2, Hv, param, grad):
#     Hlr = lr*Hv
#     dFdlr_ = dFdlr * (1-2*l2*lr) - Hlr - grad - 2*l2*param

# def update_dFdlambda_l2(self, Hv, param):
#     self.Hl2 = self.eta*Hv
#     self.Hl2_norm = torch.norm(self.Hl2)
#     self.dFdl2_norm = torch.norm(self.dFdl2)
#     self.dFdl2.data = self.dFdl2.data * (1-2*self.lambda_l2*self.eta) - self.Hl2  - 2*self.eta*param

# def update_eta(self, mlr, val_grad):
#     val_grad = torch.nn.utils.parameters_to_vector(val_grad)
#     delta = val_grad.dot(self.dFdlr).data.item()
#     # wandb.log({"dFdlr_dot_val_grad": delta}, commit=False)
#     # delta = torch.dot(val_grad, self.dFdlr).item()
#     self.eta -= mlr * delta
#     self.eta = max(0.0, self.eta)

# def update_lambda(self, mlr, val_grad):
#     val_grad = torch.nn.utils.parameters_to_vector(val_grad)
#     delta = val_grad.dot(self.dFdl2).data.item()
#     self.lambda_l2 -= mlr * delta 
#     self.lambda_l2 = max(0, self.lambda_l2)

@dataclass(frozen=True)
class RNN():
    def __init__(self, config: RnnConfig, lr_init, lambda_l2):
        self.config = config
        # self.rnn = nn.RNN(config.n_in, config.n_h, config.num_layers, batch_first=True)
        # self.fc = nn.Linear(config.n_h, config.n_out)
        
        # gru_init(self.rnn, self.fc, config)

        # with torch.no_grad():
        #     W_rec, W_in, b_rec, W_out, b_out = initializeParameters(config.n_in, config.n_h, config.n_out)
        #     self.rnn.weight_ih_l0.copy_(W_in)
        #     self.rnn.weight_hh_l0.copy_(W_rec)
        #     self.rnn.bias_ih_l0.copy_(b_rec)
        #     self.rnn.bias_hh_l0.copy_(b_rec)
        #     self.fc.weight.copy_(W_out)
        #     self.fc.bias.copy_(b_out)

        # self.interface = RNNInterface(
        #     baseCase=getRNNInit(config.scheme)(config.num_layers, config.n_h),
        #     forward=lambda x, s0: self.rnn(x, s0)[0]
        # )

        self.param_sizes = [p.numel() for p in self.parameters()]
        self.n_params = sum(self.param_sizes)
        self.param_shapes = [tuple(p.shape) for p in self.parameters()]
        self.param_cumsum = torch.cumsum(torch.tensor([0] + self.param_sizes), 0)

        self.reset_jacob()
        self.eta  = lr_init
        self.lambda_l2 = lambda_l2
        self.grad_norm = 0
        self.grad_norm_vl = 0
        self.grad_angle = 0
        self.param_norm = 0
        self.dFdlr_norm = 0
        self.dFdl2_nrom = 0

        self.getInitialActivation = getRNNInit(config.scheme, config.num_layers, config.n_h)

    
    def reset_jacob(self):
        self.dFdlr = torch.zeros(self.n_params) 
        self.dFdl2 = torch.zeros(self.n_params)
        self.dFdl2_norm = 0
        self.dFdlr_norm = 0
        
    def forward(self, x, s0):
        # s0 = self.getInitialActivation(x)
        out = self.rnn(x, s0)[0]
        activations = out.clone().detach()
        self.activations = activations
        out = self.fc(out)
        return out, activations

# Fully connected neural network with one hidden layer
class RNN(nn.Module):
    def __init__(self, config: RnnConfig, lr_init, lambda_l2):
        super(RNN, self).__init__()

        self.config = config
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

        # self.interface = RNNInterface(
        #     baseCase=getRNNInit(config.scheme)(config.num_layers, config.n_h),
        #     forward=lambda x, s0: self.rnn(x, s0)[0]
        # )

        self.param_sizes = [p.numel() for p in self.parameters()]
        self.n_params = sum(self.param_sizes)
        self.param_shapes = [tuple(p.shape) for p in self.parameters()]
        self.param_cumsum = torch.cumsum(torch.tensor([0] + self.param_sizes), 0)

        self.reset_jacob()
        self.eta  = lr_init
        self.lambda_l2 = lambda_l2
        self.grad_norm = 0
        self.grad_norm_vl = 0
        self.grad_angle = 0
        self.param_norm = 0
        self.dFdlr_norm = 0
        self.dFdl2_nrom = 0

        # self.getInitialActivation = getRNNInit(config.scheme, config.num_layers, config.n_h)

    
    def reset_jacob(self):
        self.dFdlr = torch.rand(self.n_params) - 0.5
        self.dFdl2 = torch.zeros(self.n_params)
        self.dFdl2_norm = 0
        self.dFdlr_norm = 0
        
    def forward(self, x, s0):
        # s0 = self.getInitialActivation(x)
        out = self.rnn(x, s0)[0]
        activations = out.clone().detach()
        self.activations = activations
        out = self.fc(out)
        return out, activations

    def update_dFdlr(self, Hv, param, grad):
        self.Hlr = self.eta*Hv
        self.Hlr_norm = torch.norm(self.Hlr, p=2)
        self.dFdlr_norm = torch.norm(self.dFdlr, p=2)
        self.dFdlr.data = self.dFdlr.data * (1-2*self.lambda_l2*self.eta) - self.Hlr - grad - 2*self.lambda_l2*param
        # wandb.log({"dFdlr_hv_norm": torch.norm(Hv)
        #         , "dFdlr_lr": self.eta
        #         , "Hlr_norm": self.Hlr_norm
        #         , "dFdlr_norm": self.dFdlr_norm
        #         , "dFdlr_norm_vl": self.grad_norm_vl
        #         , "dFdlr_coeff": (1-2*self.lambda_l2*self.eta)
        #         , "dFdlr_coeff_norm": torch.norm(self.dFdlr.data * (1-2*self.lambda_l2*self.eta), p=2)
        #         , "dFdlr_param_norm": torch.norm(param, p=2)
        #         , "dFdlr_grad_norm": torch.norm(grad, p=2)
        #         , "dFdlr_coeff_param_norm": torch.norm(2*self.lambda_l2*param, p=2)
        #         }, commit=False)

    def update_dFdlambda_l2(self, Hv, param):
        self.Hl2 = self.eta*Hv
        self.Hl2_norm = torch.norm(self.Hl2)
        self.dFdl2_norm = torch.norm(self.dFdl2)
        self.dFdl2.data = self.dFdl2.data * (1-2*self.lambda_l2*self.eta) - self.Hl2  - 2*self.eta*param
    
    def update_eta(self, mlr, val_grad):
        val_grad = torch.nn.utils.parameters_to_vector(val_grad)
        delta = val_grad.dot(self.dFdlr).data.item()
        # wandb.log({"dFdlr_dot_val_grad": delta}, commit=False)
        # delta = torch.dot(val_grad, self.dFdlr).item()
        self.eta -= mlr * delta
        self.eta = max(0.0, self.eta)

    def update_lambda(self, mlr, val_grad):
        val_grad = torch.nn.utils.parameters_to_vector(val_grad)
        delta = val_grad.dot(self.dFdl2).data.item()
        self.lambda_l2 -= mlr * delta 
        self.lambda_l2 = max(0, self.lambda_l2)
        # self.lambda_l2 = min(0.0002, self.lambda_l2)


def getRNNInit(initScheme: InitType, num_layers: int, n_h: int) -> Callable[[int], torch.Tensor]:
    match initScheme:
        case ZeroInit():
            return lambda batch_size: torch.zeros(num_layers, batch_size, n_h)
        case RandomInit():
            return lambda batch_size: torch.rand(num_layers, batch_size, n_h) - 0.5
        case StaticRandomInit():
            closured = torch.rand(num_layers, 1, n_h) - 0.5
            return lambda batch_size: closured.repeat(1, batch_size, 1)
        case _:
            raise Exception("Invalid init scheme type")