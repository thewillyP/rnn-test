# %%
import time

import jax.experimental
import optax
from recurrent.datarecords import InputOutput
from recurrent.monad import foldM
from recurrent.mylearning import *
from recurrent.myrecords import RnnGodState
from recurrent.mytypes import *

from recurrent.objectalgebra.interpreter import (
    BaseRnnInterpreter,
    OhoInterpreter,
    OhoRnnTrainInterpreter,
    OhoRnnValidationInterpreter,
)

from matplotlib import pyplot as plt
from recurrent.parameters import (
    RfloConfig,
    RnnConfig,
    RnnParameter,
    SgdParameter,
    UORO_Param,
)
from memory_profiler import profile
from torch.utils.data import TensorDataset, DataLoader
from toolz.itertoolz import partition_all
from recurrent.util import *
import jax.numpy as jnp
import equinox as eqx
import jax

"""
Todo
1) implement vanilla rnn training loop 
2) implement oho to show how easy it is
3) implement feedforward to show how easy it is
"""


def generate_add_task_dataset(N, t_1, t_2, tau_task, rng_key):
    """y(t) = 0.5 + 0.5 * x(t - t_1) - 0.25 * x(t - t_2)"""
    N = N // tau_task

    x = jax.random.bernoulli(rng_key, 0.5, (N,)).astype(jnp.float32)

    y = 0.5 + 0.5 * jnp.roll(x, t_1) - 0.25 * jnp.roll(x, t_2)

    X = jnp.array([x, 1 - x]).T
    Y = jnp.array([y, 1 - y]).T

    # Temporally stretch according to the desire timescale of change.
    X = jnp.tile(X, tau_task).reshape((tau_task * N, 2))
    Y = jnp.tile(Y, tau_task).reshape((tau_task * N, 2))

    return X, Y


def constructRnnEnv(rng_key: Array):
    # hard code these guys for now
    n_h = 32
    n_in = 2
    n_out = 2
    # parameters should never be floats or single value
    learning_rate = jnp.array([0.0001])
    meta_learning_rate = jnp.array([0.0001])
    alpha = 1.0

    rng_key, prng = jax.random.split(rng_key)
    w_rec_, _ = jnp.linalg.qr(jax.random.normal(prng, (n_h, n_h)))
    rng_key, prng = jax.random.split(rng_key)
    w_in_ = jax.random.normal(prng, (n_h, n_in + 1)) * jnp.sqrt(1.0 / (n_in + 1))
    w_rec = jnp.concat((w_rec_, w_in_), axis=1)
    rng_key, prng = jax.random.split(rng_key)
    w_out = jax.random.normal(prng, (n_out, n_h + 1)) * jnp.sqrt(1.0 / (n_h + 1))

    w_rec = jnp.ravel(w_rec)
    w_out = jnp.ravel(w_out)

    parameter = RnnParameter(
        w_rec=PARAMETER(w_rec),
        w_out=PARAMETER(w_out),
    )
    rnnConfig = RnnConfig(
        n_h=n_h, n_in=n_in, n_out=n_out, alpha=alpha, activationFn=jnp.tanh
    )
    sgd = SgdParameter(learning_rate=LEARNING_RATE(learning_rate))
    sgd_mlr = SgdParameter(learning_rate=LEARNING_RATE(meta_learning_rate))

    rng_key, prng1 = jax.random.split(rng_key)
    rng_key, prng2 = jax.random.split(rng_key)
    rng_key, prng3 = jax.random.split(rng_key)

    activation = ACTIVATION(jax.random.normal(prng1, (n_h,)))

    initEnv = RnnGodState[RnnParameter, SgdParameter, SgdParameter](
        # activation=ACTIVATION(torch.randn(batch_size, n_h)),
        # activation=ACTIVATION(torch.randn(n_h)),
        activation=activation,
        influenceTensor=Gradient[RnnParameter](
            eqx.filter_jacrev(lambda _: activation)(parameter)
        ),
        ohoInfluenceTensor=Gradient(eqx.filter_jacrev(lambda _: parameter)(sgd)),
        parameter=parameter,
        hyperparameter=sgd,
        metaHyperparameter=sgd_mlr,
        rnnConfig=rnnConfig,
        rnnConfig_bilevel=rnnConfig,
        rfloConfig=RfloConfig(rflo_alpha=alpha),
        rfloConfig_bilevel=RfloConfig(rflo_alpha=alpha),
        uoro=UORO_Param[RnnParameter](
            A=jax.random.normal(prng2, (n_h,)),
            B=uoroBInit(parameter, prng3),
        ),
        prng=rng_key,
        logs=Logs(loss=jnp.array(0, dtype=jnp.float32)),
        oho_logs=Logs(loss=jnp.array(0, dtype=jnp.float32)),
    )

    return rng_key, initEnv


# @profile
def mainLoop(
    dataloader: Iterable[Traversable[InputOutput]],
    t_series_length: int,
    trunc_length: int,
):

    type DL = BaseRnnInterpreter[RnnParameter, SgdParameter, SgdParameter]
    type ENV = RnnGodState[RnnParameter, SgdParameter, SgdParameter]

    rng_key = jax.random.key(0)
    rng_key, prng = jax.random.split(rng_key)
    rng_key, initEnv = constructRnnEnv(prng)

    rtrl = RTRL[
        DL,
        InputOutput,
        RnnGodState[RnnParameter, SgdParameter, SgdParameter],
        ACTIVATION,
        RnnParameter,
        jax.Array,
        PREDICTION,
    ]()
    # lambda key, shape: jax.random.uniform(key, shape, minval=-1.0, maxval=1.0)

    onlineLearner = rtrl.onlineLearning(
        doRnnStep(),
        doRnnReadout(),
        lambda a, b: LOSS(optax.safe_softmax_cross_entropy(a, b)),
    )

    onlineLearner_folded = foldRnnLearner(onlineLearner, initEnv.parameter)

    rnnLearner = endowAverageGradients(
        onlineLearner_folded, trunc_length, t_series_length
    )
    learner = rnnLearner.rnnWithGradient.flat_map(doSgdStep)
    # for online, do foldM(rnnLearner.rnnWithGradient.flat_map(doSgdStep)) and have dataloader load entire on first

    dialect = BaseRnnInterpreter[RnnParameter, SgdParameter, SgdParameter]()

    lossFn = eqx.filter_jit(rnnLearner.rnnWithLoss.func)

    for time_series in dataloader:
        # start = time.time()

        final_env = trainStep(learner, dialect, time_series, initEnv)
        initEnv = final_env
        loss, _ = lossFn(dialect, time_series, final_env)
        print(loss / t_series_length)
        # print(time.time() - start)


def ohoLoop(
    dataloader: Iterable[Traversable[OhoInputOutput]],
    t_series_length: int,
    trunc_length: int,
):

    type Train_Dl = OhoRnnTrainInterpreter[RnnParameter, SgdParameter, SgdParameter]
    type Valid_Dl = OhoRnnValidationInterpreter[
        RnnParameter, SgdParameter, SgdParameter
    ]
    type OHO = OhoInterpreter[RnnParameter, SgdParameter, SgdParameter]
    type ENV = RnnGodState[RnnParameter, SgdParameter, SgdParameter]

    rng_key = jax.random.key(0)
    rng_key, prng = jax.random.split(rng_key)
    rng_key, initEnv = constructRnnEnv(prng)

    rtrl = RFLO[
        Train_Dl | Valid_Dl,
        OhoInputOutput,
        ENV,
        ACTIVATION,
        RnnParameter,
        jax.Array,
        PREDICTION,
    ]()
    # lambda key, shape: jax.random.uniform(key, shape, minval=-1.0, maxval=1.0)

    onlineLearner = rtrl.onlineLearning(
        doRnnStep(),
        doRnnReadout(),
        lambda a, b: LOSS(optax.safe_softmax_cross_entropy(a, b)),
    )

    onlineLearner_folded = foldRnnLearner(onlineLearner, initEnv.parameter)

    rnnLearner = endowAverageGradients(
        onlineLearner_folded, trunc_length, t_series_length
    )

    trainDialect = OhoRnnTrainInterpreter[RnnParameter, SgdParameter, SgdParameter]()
    validationDialect = OhoRnnValidationInterpreter[
        RnnParameter, SgdParameter, SgdParameter
    ]()
    dialect = OhoInterpreter[RnnParameter, SgdParameter, SgdParameter](trainDialect)

    oho_rtrl = RTRL[
        OHO,
        OhoInputOutput,
        ENV,
        RnnParameter,
        SgdParameter,
        jax.Array,
        Traversable[PREDICTION],
    ]()

    def compute_loss(label: jax.Array, prediction: Traversable[PREDICTION]):
        return LOSS(optax.safe_softmax_cross_entropy(label, prediction.value))

    oho: RnnLibrary[
        OHO, Traversable[OhoInputOutput], ENV, Traversable[PREDICTION], SgdParameter
    ]
    oho = endowOho(
        rnnLearner, doSgdStep, trainDialect, validationDialect, oho_rtrl, compute_loss
    )

    learner = oho.rnnWithGradient.flat_map(doSgdStep)

    lossFn = eqx.filter_jit(rnnLearner.rnnWithLoss.func)

    for time_series in dataloader:
        # start = time.time()
        final_env = trainStep(learner, dialect, time_series, initEnv)
        initEnv = final_env
        loss, _ = lossFn(dialect, time_series, final_env)
        print(loss / t_series_length)
        # print(time.time() - start)


def onlineLearnerLoop(
    dataloader: Traversable[InputOutput],
    t_series_length: int,
):

    type DL = BaseRnnInterpreter[RnnParameter, SgdParameter, SgdParameter]
    type ENV = RnnGodState[RnnParameter, SgdParameter, SgdParameter]

    rng_key = jax.random.key(0)
    rng_key, prng = jax.random.split(rng_key)
    rng_key, initEnv = constructRnnEnv(prng)

    rtrl = UORO[
        DL,
        InputOutput,
        RnnGodState[RnnParameter, SgdParameter, SgdParameter],
        ACTIVATION,
        RnnParameter,
        jax.Array,
        PREDICTION,
    ](lambda key, shape: jax.random.uniform(key, shape, minval=-1.0, maxval=1.0))
    # lambda key, shape: jax.random.uniform(key, shape, minval=-1.0, maxval=1.0)

    onlineLearner = rtrl.onlineLearning(
        doRnnStep(),
        doRnnReadout(),
        lambda a, b: LOSS(optax.safe_softmax_cross_entropy(a, b)),
    )

    learner = onlineLearner.rnnWithGradient.flat_map(doSgdStep)

    losses = []

    def appendLoss(loss):
        losses.append(loss / t_series_length)
        # print(loss / t_series_length)

    @do()
    def next():
        print("recompiled")
        e = yield from ProxyS[ENV].get()
        dl = yield from ProxyDl[DL].askDl()
        loss, _ = accumulate(onlineLearner.rnnWithLoss, add, 0).func(dl, dataloader, e)
        jax.experimental.io_callback(appendLoss, None, loss)

        return onlineLearner.rnnWithGradient.flat_map(doSgdStep)

    dialect = BaseRnnInterpreter[RnnParameter, SgdParameter, SgdParameter]()

    # lossFn = eqx.filter_jit(accumulate(onlineLearner.rnnWithLoss, add, 0).func)

    # .lower((dialect, dataloader, initEnv))
    # .compile()
    start = time.time()
    model = (
        eqx.filter_jit(repeatM(next()).func)
        .lower(dialect, dataloader, initEnv)
        .compile()
    )
    jax.block_until_ready(model(dialect, dataloader, initEnv))
    print(f"Train Time: {time.time() - start}")

    # for time_series in map(
    #     lambda x: InputOutput(x[0], x[1]), zip(dataloader.value.x, dataloader.value.y)
    # ):
    #     # start = time.time()
    #     loss, _ = lossFn(dialect, dataloader, initEnv)
    #     losses.append(loss / t_series_length)
    #     print(loss / t_series_length)
    #     final_env = trainStep(learner, dialect, time_series, initEnv)
    #     initEnv = final_env

    plt.figure(figsize=(10, 6))
    plt.plot(losses, linestyle="-", color="b", label="Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # Save the plot to disk
    output_path = "test2.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")


def main2():
    N = 10_000
    rng_key = jax.random.key(0)
    rng_key, prng1, prng2 = jax.random.split(rng_key, 3)
    X, Y = generate_add_task_dataset(N, 5, 9, 1, prng1)

    dataloader = Traversable(InputOutput(x=X, y=Y))

    onlineLearnerLoop(dataloader, N)


def main():

    N = 100_000
    N_val = 10_000
    t_series_length = 100  # how much time series goines into ONE param update
    trunc_length = 1  # controls how much avging done in one t_series
    # if trunc_length = 1, then divide by t_series_length. if trunc_length = t_series_length, then no normalization done
    rng_key = jax.random.key(0)
    rng_key, prng1, prng2 = jax.random.split(rng_key, 3)
    X, Y = generate_add_task_dataset(N, 5, 9, 1, prng1)
    X_val, Y_val = generate_add_task_dataset(N_val, 5, 9, 1, prng2)

    def transform(arr: Array, _t: int):
        return arr.reshape((_t, -1) + arr.shape[1:])

    X = transform(X, N // t_series_length)
    Y = transform(Y, N // t_series_length)
    # X_val = transform(X_val, N // t_series_length)
    # Y_val = transform(Y_val, N // t_series_length)

    dataloader = map(
        lambda data: Traversable(
            OhoInputOutput(
                train=InputOutput(x=data[0], y=data[1]),
                val=InputOutput(x=X_val, y=Y_val),
            )
        ),
        zip(X, Y),
    )

    ohoLoop(dataloader, t_series_length, trunc_length)

    # dataloader = map(
    #     lambda data: Traversable[InputOutput](InputOutput(data[0], data[1])), zip(X, Y)
    # )

    # mainLoop(dataloader, t_series_length, trunc_length)

    # dataset = map(lambda data: InputOutput(data[0], data[1]), zip(X, Y))
    # dataset = list(dataset)

    # import cProfile, pstats

    # profiler = cProfile.Profile()
    # profiler.enable()
    # start = time.time()
    # predictions = temp(dataset)
    # print(time.time() - start)
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats("cumtime")
    # stats.print_stats()
    # stats.dump_stats("profile_results.prof")

    # predictions = list(predictions)
    # indices = torch.arange(len(predictions))

    # predictions = [tensor[0].item() for tensor in predictions]
    # labels = [tensor[0].item() for tensor in Y]

    # # Plot the data
    # plt.figure(figsize=(8, 5))
    # plt.plot(indices, predictions, marker="o", label="Prediction")
    # plt.plot(indices, labels, marker="o", label="Target")
    # plt.title("Plot of List Data with Indices")
    # plt.xlabel("Indices")
    # plt.ylabel("Values")
    # plt.grid(True)
    # plt.legend()
    # plt.show()

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
