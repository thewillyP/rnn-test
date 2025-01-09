from recurrent.mylearning import createRnnLibrary
from recurrent.myrecords import RnnBPTTState, WithRnnConfig
import torch
from recurrent.mytypes import *

from recurrent.objectalgebra.base_interpreter import RnnLearnableInterpreter
from recurrent.objectalgebra.tensor_intepreter import Input2Output1Interpreter

"""
Todo
1) implement vanilla rnn training loop
2) implement feedforward to show how easy it is
3) implement oho to show how easy it is
"""


def rnnStep(
    config: WithRnnConfig, x: torch.Tensor, a: torch.Tensor, w_rec: torch.Tensor
) -> torch.Tensor:
    n_h, n_in = config.n_h, config.n_in
    w_rec = torch.reshape(w_rec, (n_h, n_h + n_in + 1))
    return (1 - config.alpha) * a + config.alpha * config.activationFn(
        w_rec @ torch.cat((x, a, torch.tensor([1.0])))
    )


def rnnReadout(
    config: WithRnnConfig, _: torch.Tensor, a: torch.Tensor, w_out: torch.Tensor
) -> torch.Tensor:
    w_out = torch.reshape(w_out, (config.n_out, config.n_h + 1))
    return w_out @ torch.cat((a, torch.tensor([1.0])))


def trainStep(batch_size: int):

    n_h = 30
    n_in = 2
    n_out = 1
    learning_rate = 0.1
    alpha = 1.0

    dialect = RnnLearnableInterpreter[RnnBPTTState]
    dataDialect = Input2Output1Interpreter
    parameters = torch.randn(n_h * (n_h + n_in + 1) + n_out * (n_h + 1))

    initEnv: RnnBPTTState = RnnBPTTState(
        activation=ACTIVATION(torch.randn(batch_size, n_h)),
        parameter=PARAMETER(parameters),
        hyperparameter=HYPERPARAMETER(torch.tensor(learning_rate)),
        n_h=n_h,
        n_in=n_in,
        n_out=n_out,
        alpha=alpha,
        activationFn=torch.tanh,
    )

    rnnLibrary = createRnnLibrary(
        dialect,
        dataDialect,
        lambda x, a, w: ACTIVATION(rnnStep(initEnv, x, a, w)),
        lambda _, a, w: PREDICTION(rnnReadout(initEnv, _, a, w)),
    )

    print(rnnLibrary)
