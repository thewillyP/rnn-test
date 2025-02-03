# %%
import time

import jax.experimental
import optax
from recurrent.datarecords import InputOutput, OhoInputOutput
from recurrent.monad import foldM
from recurrent.mylearning import *
from recurrent.myrecords import RnnGodState
from recurrent.mytypes import *

from recurrent.objectalgebra.interpreter import (
    BaseRnnInterpreter,
    OhoInterpreter,
)

from matplotlib import pyplot as plt
from recurrent.parameters import (
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

    X = jnp.asarray([x, 1 - x]).T
    Y = jnp.asarray([y, 1 - y]).T

    # Temporally stretch according to the desire timescale of change.
    X = jnp.tile(X, tau_task).reshape((tau_task * N, 2))
    Y = jnp.tile(Y, tau_task).reshape((tau_task * N, 2))

    return X, Y


def constructRnnEnv(rng_key: Array):
    """Constructs an RNN environment with predefined configurations."""

    # Define network dimensions
    n_h, n_in, n_out = 32, 2, 2
    alpha = 1.0

    # Define learning rates as arrays
    learning_rate = jnp.asarray([0.01])
    meta_learning_rate = jnp.asarray([0.001])

    # Generate random weights
    def random_matrix(rng_key, shape, scale):
        rng_key, prng = jax.random.split(rng_key)
        return rng_key, jax.random.normal(prng, shape) * scale

    rng_key, prng = jax.random.split(rng_key)
    w_rec_, _ = jnp.linalg.qr(jax.random.normal(prng, (n_h, n_h)))

    rng_key, w_in_ = random_matrix(rng_key, (n_h, n_in + 1), jnp.sqrt(1.0 / (n_in + 1)))
    w_rec = jnp.concatenate((w_rec_, w_in_), axis=1)

    rng_key, w_out = random_matrix(rng_key, (n_out, n_h + 1), jnp.sqrt(1.0 / (n_h + 1)))

    # Initialize parameters
    parameter = RnnParameter(w_rec=w_rec, w_out=w_out)
    rnn_config = RnnConfig(n_h=n_h, n_in=n_in, n_out=n_out, alpha=alpha, activationFn=jnp.tanh)

    sgd = SgdParameter(learning_rate=learning_rate)
    sgd_mlr = SgdParameter(learning_rate=meta_learning_rate)

    # Generate random activation state
    rng_key, activation_rng = jax.random.split(rng_key)
    activation = ACTIVATION(jax.random.normal(activation_rng, (n_h,)))

    # Split keys for UORO
    rng_key, prng_A, prng_B, prng_C = jax.random.split(rng_key, num=4)

    # Construct environment state
    init_env = RnnGodState[RnnParameter, SgdParameter, SgdParameter](
        activation=activation,
        influenceTensor=Jacobian[RnnParameter](zeroedInfluenceTensor(n_h, parameter)),
        ohoInfluenceTensor=Jacobian[SgdParameter](
            zeroedInfluenceTensor(jnp.size(compose2(endowVector, toVector)(parameter)), sgd)
        ),
        parameter=parameter,
        hyperparameter=sgd,
        metaHyperparameter=sgd_mlr,
        rnnConfig=rnn_config,
        rnnConfig_bilevel=rnn_config,
        uoro=UORO_Param(
            A=jax.random.normal(prng_A, (n_h,)),
            B=uoroBInit(prng_B, parameter),
        ),
        prng=prng_C,
    )

    return rng_key, init_env


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
        InputOutput,
        ENV,
        ACTIVATION,
        RnnParameter,
        jax.Array,
        PREDICTION,
    ]()
    # lambda key, shape: jax.random.uniform(key, shape, minval=-1.0, maxval=1.0)

    onlineLearner: RnnLibrary[DL, InputOutput, ENV, PREDICTION, RnnParameter]
    onlineLearner = rtrl.createLearner(
        doRnnStep(),
        doRnnReadout(),
        lambda a, b: LOSS(optax.safe_softmax_cross_entropy(a, b)),
        readoutRecurrentError(doRnnReadout(), lambda a, b: LOSS(optax.safe_softmax_cross_entropy(a, b))),
    )

    onlineLearner_folded = foldrRnnLearner(onlineLearner, initEnv.parameter)

    rnnLearner = endowAveragedGradients(onlineLearner_folded, trunc_length, initEnv.parameter)
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
    dataloader: Traversable[OhoInputOutput],
    t_series_length: int,
    trunc_length: int,
    total_steps: int,
    test_set: Traversable[InputOutput],
    total_test_steps: int,
):
    type Train_Dl = BaseRnnInterpreter[RnnParameter, SgdParameter, SgdParameter]
    type OHO = OhoInterpreter[RnnParameter, SgdParameter, SgdParameter]
    type ENV = RnnGodState[RnnParameter, SgdParameter, SgdParameter]

    rng_key = jax.random.key(0)
    rng_key, prng = jax.random.split(rng_key)
    rng_key, initEnv = constructRnnEnv(prng)

    rtrl = RTRL[
        InputOutput,
        ENV,
        ACTIVATION,
        RnnParameter,
        jax.Array,
        PREDICTION,
    ]()
    # lambda key, shape: jax.random.uniform(key, shape, minval=-1.0, maxval=1.0)

    onlineLearner: RnnLibrary[Train_Dl, InputOutput, ENV, PREDICTION, RnnParameter]
    onlineLearner = rtrl.createLearner(
        doRnnStep(),
        doRnnReadout(),
        lambda a, b: LOSS(optax.safe_softmax_cross_entropy(a, b)),
        readoutRecurrentError(doRnnReadout(), lambda a, b: LOSS(optax.safe_softmax_cross_entropy(a, b))),
    )

    onlineLearner_folded = foldrRnnLearner(onlineLearner, initEnv.parameter)

    rnnLearner = endowAveragedGradients(onlineLearner_folded, trunc_length, initEnv.parameter)

    trainDialect = BaseRnnInterpreter[RnnParameter, SgdParameter, SgdParameter]()
    dialect = OhoInterpreter[RnnParameter, SgdParameter, SgdParameter](trainDialect)

    oho_rtrl = RTRL[
        OhoInputOutput,
        ENV,
        RnnParameter,
        SgdParameter,
        jax.Array,
        Traversable[PREDICTION],
    ]()

    def compute_loss(label: Traversable[jax.Array], prediction: Traversable[PREDICTION]):
        return LOSS(jnp.mean(optax.safe_softmax_cross_entropy(label.value, prediction.value)))

    oho: RnnLibrary[OHO, OhoInputOutput, ENV, Traversable[PREDICTION], SgdParameter]
    oho = endowBilevelOptimization(
        rnnLearner, doSgdStep_Softplus, trainDialect, oho_rtrl, compute_loss, resetRnnActivation(initEnv.activation)
    )

    lossFn = eqx.filter_jit(accumulate(rnnLearner.rnnWithLoss, add, 0).func)

    @do()
    def log():
        env = yield from get(PX[ENV]())
        interpreter = yield from askForInterpreter(PX[OHO]())
        oho_data = yield from ask(PX[OhoInputOutput]())
        learning_rate = yield from interpreter.getParameter().fmap(lambda x: x.learning_rate)
        train_loss, _ = rnnLearner.rnnWithLoss.func(trainDialect, oho_data.train, env)
        # lag 1 timestep behind bc show prev param validation
        validation_loss, _ = rnnLearner.rnnWithLoss.func(trainDialect, oho_data.validation, env)
        test_loss, _ = rnnLearner.rnnWithLoss.func(trainDialect, test_set, env)

        log = Logs(
            train_loss=train_loss,
            validation_loss=validation_loss,
            test_loss=test_loss / total_test_steps,
            learning_rate=learning_rate,
            effective_learning_rate=jnp.maximum(0, learning_rate),
        )
        return pure(log, PX3[OHO, OhoInputOutput, ENV]())

    model = (
        eqx.filter_jit(traverse(oho.rnnWithGradient.flat_map(doSgdStep).then(log())).func)
        .lower(dialect, dataloader, initEnv)
        .compile()
    )

    # model2 = eqx.filter_jit(repeatM(onlineLearner.rnn).func).lower(dialect, dataloader, initEnv).compile()

    start = time.time()
    logs, trained_env = model(dialect, dataloader, initEnv)
    jax.block_until_ready(trained_env)
    print(f"Train Time: {time.time() - start}")
    logs = logs.value

    eqx.tree_serialise_leaves("some_filename.eqx", logs)
    print(logs.learning_rate)
    # print(logs.learning_rate)
    print(logs.test_loss)

    # loss, _ = lossFn(trainDialect, Traversable(dataloader.value.train), trained_env)
    # print(loss / total_steps)
    # print(trained_env.hyperparameter.learning_rate)

    # test_loss, _ = lossFn(trainDialect, test_set, trained_env)
    # print(test_loss / total_test_steps)

    # learner = oho.rnnWithGradient.flat_map(doSgdStep)

    # lossFn = eqx.filter_jit(rnnLearner.rnnWithLoss.func)

    # for time_series in dataloader:
    #     # start = time.time()
    #     final_env = trainStep(learner, dialect, time_series, initEnv)
    #     initEnv = final_env
    #     loss, _ = lossFn(dialect, time_series, final_env)
    #     print(loss / t_series_length)
    #     # print(time.time() - start)


def onlineLearnerLoop(
    dataloader: Traversable[InputOutput],
    t_series_length: int,
):
    type DL = BaseRnnInterpreter[RnnParameter, SgdParameter, SgdParameter]
    type ENV = RnnGodState[RnnParameter, SgdParameter, SgdParameter]

    rng_key = jax.random.key(0)
    rng_key, prng = jax.random.split(rng_key)
    rng_key, initEnv = constructRnnEnv(prng)

    rtrl = RTRL[
        InputOutput,
        RnnGodState[RnnParameter, SgdParameter, SgdParameter],
        ACTIVATION,
        RnnParameter,
        jax.Array,
        PREDICTION,
    ]()
    # lambda key, shape: jax.random.uniform(key, shape, minval=-1.0, maxval=1.0)

    onlineLearner: RnnLibrary[DL, InputOutput, ENV, PREDICTION, RnnParameter]
    onlineLearner = rtrl.createLearner(
        doRnnStep(),
        doRnnReadout(),
        lambda a, b: LOSS(optax.safe_softmax_cross_entropy(a, b)),
        readoutRecurrentError(doRnnReadout(), lambda a, b: LOSS(optax.safe_softmax_cross_entropy(a, b))),
    )

    @do()
    def next():
        print("recompiled")
        env = yield from get(PX[ENV]())
        interpreter = yield from askForInterpreter(PX[DL])
        # loss, _ = accumulate(onlineLearner.rnnWithLoss, add, 0).func(interpreter, dataloader, env)
        return onlineLearner.rnnWithGradient.flat_map(doSgdStep)

    dialect = BaseRnnInterpreter[RnnParameter, SgdParameter, SgdParameter]()

    model = (
        eqx.filter_jit(
            lambda a1, a2, a3: repeatM(onlineLearner.rnnWithGradient.flat_map(doSgdStep)).func(a2, a1, a3),
            donate="all-except-first",
        )
        .lower(dataloader, dialect, initEnv)
        .compile()
    )

    lossFn = (
        eqx.filter_jit(accumulate(onlineLearner.rnnWithLoss, add, 0).func).lower(dialect, dataloader, initEnv).compile()
    )

    # model2 = eqx.filter_jit(repeatM(onlineLearner.rnn).func).lower(dialect, dataloader, initEnv).compile()

    start = time.time()
    _, trained_env = model(dataloader, dialect, initEnv)
    jax.block_until_ready(trained_env)
    print(f"Train Time: {time.time() - start}")

    loss, _ = lossFn(dialect, dataloader, trained_env)
    print(loss / t_series_length)

    # for time_series in map(
    #     lambda x: InputOutput(x[0], x[1]), zip(dataloader.value.x, dataloader.value.y)
    # ):
    #     # start = time.time()
    #     loss, _ = lossFn(dialect, dataloader, initEnv)
    #     losses.append(loss / t_series_length)
    #     print(loss / t_series_length)
    #     final_env = trainStep(learner, dialect, time_series, initEnv)
    #     initEnv = final_env

    # plt.figure(figsize=(10, 6))
    # plt.plot(losses, linestyle="-", color="b", label="Loss")
    # plt.title("Training Loss")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.grid(True, linestyle="--", alpha=0.6)
    # plt.legend()
    # plt.tight_layout()

    # # Save the plot to disk
    # output_path = "test2.png"
    # plt.savefig(output_path)
    # print(f"Plot saved to {output_path}")


def main2():
    N = 10_000
    rng_key = jax.random.key(0)
    rng_key, prng1, prng2 = jax.random.split(rng_key, 3)
    X, Y = generate_add_task_dataset(N, 5, 9, 1, prng1)

    dataloader = Traversable(InputOutput(x=X, y=Y))

    onlineLearnerLoop(dataloader, N)


def main():
    N = 100
    N_val = 100
    N_test = 100
    t_series_length = 1  # how much time series goines into ONE param update
    trunc_length = 1  # controls how much avging done in one t_series
    # if trunc_length = 1, then divide by t_series_length. if trunc_length = t_series_length, then no normalization done
    rng_key = jax.random.key(0)
    rng_key, prng1, prng2, prng3 = jax.random.split(rng_key, 4)
    X, Y = generate_add_task_dataset(N, 5, 9, 1, prng1)
    X_val, Y_val = generate_add_task_dataset(N_val, 5, 9, 1, prng2)
    X_test, Y_test = generate_add_task_dataset(N_test, 5, 9, 1, prng3)

    def transform(arr: Array, _t: int):
        return arr.reshape((_t, -1) + arr.shape[1:])

    numUpdates = N // t_series_length

    X = transform(X, numUpdates)
    Y = transform(Y, numUpdates)
    # for every param update, go through whole validation
    X_val = jnp.repeat(jnp.expand_dims(X_val, axis=0), numUpdates, axis=0)
    Y_val = jnp.repeat(jnp.expand_dims(Y_val, axis=0), numUpdates, axis=0)

    # X_val = transform(X_val, N // t_series_length)
    # Y_val = transform(Y_val, N // t_series_length)

    dataloader = Traversable(
        OhoInputOutput(
            train=Traversable(InputOutput(x=X, y=Y)),
            validation=Traversable(InputOutput(x=X_val, y=Y_val)),
            labels=Traversable(Y_val),
        )
    )

    test_set = Traversable(InputOutput(x=X_test, y=Y_test))

    # dataloader = Traversable(
    #     Traversable(
    #         OhoInputOutput(
    #             train=InputOutput(x=X, y=Y),
    #             val=InputOutput(x=X_val, y=Y_val),
    #         )
    #     )
    # )

    # dataloader = map(
    #     lambda data: Traversable(
    #         OhoInputOutput(
    #             train=InputOutput(x=data[0], y=data[1]),
    #             val=InputOutput(x=X_val, y=Y_val),
    #         )
    #     ),
    #     zip(X, Y),
    # )

    ohoLoop(dataloader, t_series_length, trunc_length, N, test_set, N_test)

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
