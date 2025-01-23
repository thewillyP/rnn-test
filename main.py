# %%
import time
from typing import Iterable, Iterator
from recurrent.datarecords import Input2Output1
from recurrent.monad import foldM
from recurrent.mylearning import *
from recurrent.myrecords import RnnGodState
import torch
from recurrent.mytypes import *

from recurrent.objectalgebra.interpreter import BaseRnnInterpreter

from matplotlib import pyplot as plt
from recurrent.parameters import (
    RfloConfig,
    RnnConfig,
    RnnParameter,
    SgdParameter,
    UORO_Param,
)
from recurrent.util import rnnSplitParameters, tree_stack
from memory_profiler import profile
from torch.utils.data import TensorDataset, DataLoader
from toolz.itertoolz import partition_all
from torch.utils import _pytree as pytree
from torch.utils._pytree import PyTree
from recurrent.util import *

"""
Todo
1) implement vanilla rnn training loop 
2) implement feedforward to show how easy it is
3) implement oho to show how easy it is
"""


torch.manual_seed(24)


def generate_add_task_dataset(N, t_1, t_2, deterministic, tau_task):
    """y(t) = 0.5 + 0.5 * x(t - t_1) - 0.25 * x(t - t_2)"""
    N = N // tau_task

    x = torch.bernoulli(torch.full((N,), 0.5))

    y = 0.5 + 0.5 * torch.roll(x.float(), t_1) - 0.25 * torch.roll(x.float(), t_2)

    if not deterministic:
        y = torch.bernoulli(y)

    X = torch.stack([x, 1 - x], dim=-1).repeat(1, tau_task).view(tau_task * N, 2)
    Y = torch.stack([y, 1 - y], dim=-1).repeat(1, tau_task).view(tau_task * N, 2)

    return X, Y


# @profile
def trainStep(dataloader: Iterable[Input2Output1]):

    n_h = 32
    n_in = 2
    n_out = 2
    learning_rate = torch.tensor(
        [0.0001]
    )  # parameters should never be floats or single value
    alpha = 1.0

    type DL = BaseRnnInterpreter[RnnParameter, SgdParameter, SgdParameter]
    type ENV = RnnGodState[RnnParameter, SgdParameter, SgdParameter]

    dialect = BaseRnnInterpreter[RnnParameter, SgdParameter, SgdParameter]()

    w_rec_, _ = torch.linalg.qr(torch.randn(n_h, n_h))
    w_in_ = torch.randn(n_h, n_in + 1) * torch.sqrt(torch.tensor(1.0 / (n_in + 1)))
    w_rec = torch.cat((w_rec_, w_in_), dim=1)
    w_out = torch.randn(n_out, n_h + 1) * torch.sqrt(torch.tensor(1.0 / (n_h + 1)))

    w_rec = torch.flatten(w_rec)
    w_out = torch.flatten(w_out)

    parameter = RnnParameter(
        w_rec=PARAMETER(w_rec),
        w_out=PARAMETER(w_out),
    )
    rnnConfig = RnnConfig(
        n_h=n_h, n_in=n_in, n_out=n_out, alpha=alpha, activationFn=torch.tanh
    )
    sgd = SgdParameter(learning_rate=LEARNING_RATE(learning_rate))

    initEnv = RnnGodState[RnnParameter, SgdParameter, SgdParameter](
        # activation=ACTIVATION(torch.randn(batch_size, n_h)),
        activation=ACTIVATION(torch.randn(n_h)),
        influenceTensor=Gradient[RnnParameter](zeroedInfluenceTensor(n_h, parameter)),
        ohoInfluenceTensor=Gradient[SgdParameter](
            zeroedInfluenceTensor(pytreeNumel(sgd), sgd)
        ),
        parameter=parameter,
        hyperparameter=sgd,
        metaHyperparameter=sgd,
        rnnConfig=rnnConfig,
        rnnConfig_bilevel=rnnConfig,
        rfloConfig=RfloConfig(rflo_alpha=alpha),
        rfloConfig_bilevel=RfloConfig(rflo_alpha=alpha),
        uoro=UORO_Param[RnnParameter](
            A=torch.randn(n_h),
            B=uoroBInit(parameter),
        ),
    )

    # truncation = 10
    # dataloader: Iterator[Iterable[Input2Output1]] = partition_all(
    #     truncation, dataloader
    # )
    # truncated_data = map(tree_stack, dataloader)
    # need to fold over entire dataset man

    # predictions, _ = rnnLearner.rnn.func(dialect, dataloader, initEnv)
    # gr, _ = rnnLearner.rnnWithGradient.func(dialect, dataloader, initEnv)

    # print(gr)
    # print(_)

    # rnnLearner: RnnLibrary[
    #     DL, Iterable[Input2Output1], ENV, Iterable[PREDICTION], RnnParameter
    # ]
    # print(len(dataloader))
    # rnnLearner = offlineLearning(
    #     doRnnStep(),
    #     doRnnReadout(),
    #     lambda a, b: LOSS(torch.functional.F.cross_entropy(a, b) / len(dataloader)),
    # )

    # def run(prev_env):
    #     _, env = rnnLearner.rnnWithGradient.flat_map(doSgdStep).func(
    #         dialect, dataloader, prev_env
    #     )
    #     loss, _ = rnnLearner.rnnWithLoss.func(dialect, dataloader, env)
    #     print(loss)
    #     new_env = copy.replace(prev_env, parameter=env.parameter)
    #     return new_env

    # for _ in range(200):
    #     initEnv = run(initEnv)

    # predictions, _ = rnnLearner.rnn.func(dialect, dataloader, initEnv)
    # predictions = [torch.functional.F.softmax(tensor, dim=0) for tensor in predictions]
    # return predictions

    rnnLearner: RnnLibrary[DL, Input2Output1, ENV, PREDICTION, RnnParameter]
    rtrl = RTRL[
        DL,
        Input2Output1,
        RnnGodState[RnnParameter, SgdParameter, SgdParameter],
        ACTIVATION,
        RnnParameter,
        Tensor,
        PREDICTION,
    ]()
    rnnLearner = rtrl.onlineLearning(
        doRnnStep(),
        doRnnReadout(),
        lambda a, b: LOSS(torch.functional.F.cross_entropy(a, b)),
    )
    # gr, _ = rnnLearner.rnnWithGradient.func(dialect, dataloader[0], initEnv)

    # _, initEnv = rnnLearner.trainStep(doSgdStep).func(dialect, dataloader, initEnv)
    for d in dataloader:
        model = torch.compile(rnnLearner.rnnWithGradient.flat_map(doSgdStep).func)
        _, initEnv = model(dialect, d, initEnv)
        # print(initEnv.parameter.w_out)
        # lossFn = foldM(
        #     lambda acc: rnnLearner.rnnWithLoss.fmap(lambda l: l + acc), LOSS(0)
        # )
        # loss, _ = lossFn.func(dialect, dataloader, initEnv)
        # print(loss / len(dataloader))

    predictions, _ = traverse(rnnLearner.rnn).func(dialect, dataloader, initEnv)
    predictions = [torch.functional.F.softmax(tensor, dim=0) for tensor in predictions]
    return predictions


# %%


def main():

    length = 100
    X, Y = generate_add_task_dataset(length, 5, 9, True, 1)
    dataset = map(lambda data: Input2Output1(data[0], data[1]), zip(X, Y))
    dataset = list(dataset)

    # import cProfile, pstats

    # profiler = cProfile.Profile()
    # profiler.enable()
    # start = time.time()
    predictions = trainStep(dataset)
    # print(time.time() - start)
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats("cumtime")
    # stats.print_stats()
    # stats.dump_stats("profile_results.prof")

    predictions = list(predictions)
    indices = torch.arange(len(predictions))

    predictions = [tensor[0].item() for tensor in predictions]
    labels = [tensor[0].item() for tensor in Y]

    # Plot the data
    plt.figure(figsize=(8, 5))
    plt.plot(indices, predictions, marker="o", label="Prediction")
    plt.plot(indices, labels, marker="o", label="Target")
    plt.title("Plot of List Data with Indices")
    plt.xlabel("Indices")
    plt.ylabel("Values")
    plt.grid(True)
    plt.legend()
    plt.show()

    # # Define the simple RNN model
    # class SimpleRNN(torch.nn.Module):
    #     def __init__(self, input_size, hidden_size, output_size):
    #         super(SimpleRNN, self).__init__()
    #         self.hidden_size = hidden_size
    #         # Define the RNN layer
    #         self.rnn = torch.nn.RNN(input_size, hidden_size)
    #         # Define a fully connected layer to map RNN output to final prediction
    #         self.fc = torch.nn.Linear(hidden_size, output_size)

    #     def forward(self, x):
    #         # Initialize hidden state with zeros
    #         h0 = torch.zeros(1, self.hidden_size)
    #         # Get RNN output (output, hidden_state)
    #         out, _ = self.rnn(x, h0)
    #         # Pass the RNN output through the fully connected layer for each time step
    #         out = self.fc(out)  # Apply to all time steps
    #         return out

    # random_data = torch.randn(length, 2)

    # # Initialize the model
    # input_size = 2  # Corresponding to x1, x2, and y
    # hidden_size = 30  # Arbitrary size for hidden state
    # output_size = 1  # We are predicting a single output value

    # model = SimpleRNN(input_size, hidden_size, output_size)

    # # Generate predictions for the entire sequence
    # with torch.no_grad():  # No need to compute gradients
    #     start_time_vmap = time.time()
    #     predictions = model(random_data)
    #     vmap_time = time.time() - start_time_vmap
    #     print(f"vmap execution time: {vmap_time:.6f} seconds")

    # print(predictions.shape)
    # # %%


main()

# %%
