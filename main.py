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

# jax.config.update("jax_enable_x64", True)

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


def constructRnnEnv(rng_key: Array, initLearningRate: float):
    """Constructs an RNN environment with predefined configurations."""

    # Define network dimensions
    n_h, n_in, n_out = 32, 2, 2
    alpha = 1.0

    # Define learning rates as arrays
    # 0.02712261
    learning_rate = jnp.asarray([initLearningRate])
    meta_learning_rate = jnp.asarray([0.0001])

    rng_key, subkey1, subkey2, subkey3 = jax.random.split(rng_key, 4)

    W_in = jax.random.normal(subkey1, (n_h, n_in)) * jnp.sqrt(1 / n_in)
    W_rec, _ = jnp.linalg.qr(jax.random.normal(subkey2, (n_h, n_h)))  # QR decomposition
    W_out = jax.random.normal(subkey3, (n_out, n_h)) * jnp.sqrt(1 / n_h)
    b_rec = jnp.zeros((n_h, 1))
    b_out = jnp.zeros((n_out, 1))

    w_rec = jnp.hstack([W_rec, W_in, b_rec])
    w_out = jnp.hstack([W_out, b_out])

    # # Generate random weights
    # def random_matrix(rng_key, shape, scale):
    #     rng_key, prng = jax.random.split(rng_key)
    #     return rng_key, jax.random.normal(prng, shape) * scale

    # rng_key, prng = jax.random.split(rng_key)
    # w_rec_, _ = jnp.linalg.qr(jax.random.normal(prng, (n_h, n_h)))

    # rng_key, w_in_ = random_matrix(rng_key, (n_h, n_in + 1), jnp.sqrt(1.0 / (n_in + 1)))
    # w_rec = jnp.concatenate((w_rec_, w_in_), axis=1)

    # rng_key, w_out = random_matrix(rng_key, (n_out, n_h + 1), jnp.sqrt(1.0 / (n_h + 1)))

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
    initLearningRate: float,
):
    type DL = BaseRnnInterpreter[RnnParameter, SgdParameter, SgdParameter]
    type ENV = RnnGodState[RnnParameter, SgdParameter, SgdParameter]

    rng_key = jax.random.key(0)
    rng_key, prng = jax.random.split(rng_key)
    rng_key, initEnv = constructRnnEnv(prng, initLearningRate)

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
    initLearningRate: float,
):
    type Train_Dl = BaseRnnInterpreter[RnnParameter, SgdParameter, SgdParameter]
    type OHO = OhoInterpreter[RnnParameter, SgdParameter, SgdParameter]
    type ENV = RnnGodState[RnnParameter, SgdParameter, SgdParameter]

    rng_key = jax.random.key(0)
    rng_key, prng = jax.random.split(rng_key)
    rng_key, initEnv = constructRnnEnv(prng, initLearningRate)

    rtrl = UORO[
        InputOutput,
        ENV,
        ACTIVATION,
        RnnParameter,
        jax.Array,
        PREDICTION,
    ](lambda key, shape: jax.random.uniform(key, shape, minval=-1.0, maxval=1.0))
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
        rnnLearner, doSgdStep, trainDialect, oho_rtrl, compute_loss, resetRnnActivation(initEnv.activation)
    )

    lossFn = eqx.filter_jit(accumulate(rnnLearner.rnnWithLoss, add, 0.0).func)

    @do()
    def updateStep():
        # env = yield from get(PX[ENV]())
        interpreter = yield from askForInterpreter(PX[OHO]())
        # oho_data = yield from ask(PX[OhoInputOutput]())
        learning_rate = yield from interpreter.getParameter().fmap(lambda x: x.learning_rate)
        # train_loss, _ = rnnLearner.rnnWithLoss.func(trainDialect, oho_data.train, env)
        # # lag 1 timestep behind bc show prev param validation
        # validation_loss, _ = (
        #     resetRnnActivation(initEnv.activation)
        #     .then(rnnLearner.rnnWithLoss)
        #     .func(trainDialect, oho_data.validation, env)
        # )
        # # test_loss, _ = (
        # #     resetRnnActivation(initEnv.activation).then(rnnLearner.rnnWithLoss).func(trainDialect, test_set, env)
        # # )
        # log = Logs(
        #     train_loss=train_loss,
        #     validation_loss=validation_loss,
        #     test_loss=None,
        #     learning_rate=learning_rate,
        # )
        log = Logs(
            train_loss=None,
            validation_loss=None,
            test_loss=None,
            learning_rate=learning_rate,
        )
        _ = yield from oho.rnnWithGradient.flat_map(doSgdStep_Clipped)

        return pure(log, PX3[OHO, OhoInputOutput, ENV]())

    model = eqx.filter_jit(traverse(updateStep()).func).lower(dialect, dataloader, initEnv).compile()

    # model2 = eqx.filter_jit(repeatM(onlineLearner.rnn).func).lower(dialect, dataloader, initEnv).compile()

    start = time.time()
    logs, trained_env = model(dialect, dataloader, initEnv)
    tttt = f"Train Time: {time.time() - start}"
    logs = logs.value

    train_loss, _ = lossFn(trainDialect, Traversable(dataloader.value.train), trained_env)
    test_loss, _ = eqx.filter_jit(rnnLearner.rnnWithLoss.func)(trainDialect, test_set, trained_env)

    train_loss = train_loss / total_steps
    test_loss = test_loss / total_test_steps
    print(train_loss)
    print(test_loss)
    print(logs.learning_rate[-1])
    print(tttt)

    return logs, test_loss

    # eqx.tree_serialise_leaves("some_filename.eqx", logs)
    # print(logs.learning_rate**2)
    # # print(logs.learning_rate)
    # print(logs.test_loss)


def onlineLearnerLoop(
    dataloader: Traversable[InputOutput],
    t_series_length: int,
    test_set: Traversable[InputOutput],
    test_steps: int,
    initLearningRate: float,
):
    type DL = BaseRnnInterpreter[RnnParameter, SgdParameter, SgdParameter]
    type ENV = RnnGodState[RnnParameter, SgdParameter, SgdParameter]

    rng_key = jax.random.key(0)
    rng_key, prng = jax.random.split(rng_key)
    rng_key, initEnv = constructRnnEnv(prng, initLearningRate)

    rtrl = UORO[
        InputOutput,
        RnnGodState[RnnParameter, SgdParameter, SgdParameter],
        ACTIVATION,
        RnnParameter,
        jax.Array,
        PREDICTION,
    ](lambda key, shape: jax.random.uniform(key, shape, minval=-1.0, maxval=1.0))
    # lambda key, shape: jax.random.uniform(key, shape, minval=-1.0, maxval=1.0)

    onlineLearner: RnnLibrary[DL, InputOutput, ENV, PREDICTION, RnnParameter]
    onlineLearner = rtrl.createLearner(
        doRnnStep(),
        doRnnReadout(),
        lambda a, b: LOSS(optax.safe_softmax_cross_entropy(a, b)),
        readoutRecurrentError(doRnnReadout(), lambda a, b: LOSS(optax.safe_softmax_cross_entropy(a, b))),
    )

    dialect = BaseRnnInterpreter[RnnParameter, SgdParameter, SgdParameter]()

    lossFn = eqx.filter_jit(
        resetRnnActivation(initEnv.activation).then(accumulate(onlineLearner.rnnWithLoss, add, 0.0)).func
    )

    @do()
    def updateStep():
        env = yield from get(PX[ENV]())
        data = yield from ask(PX[InputOutput]())
        train_loss, _ = onlineLearner.rnnWithLoss.func(dialect, data, env)
        # test_loss, _ = lossFn(dialect, test_set, env)
        log = Logs(
            train_loss=train_loss,
            validation_loss=None,
            test_loss=None,
            learning_rate=None,
        )
        _ = yield from onlineLearner.rnnWithGradient.flat_map(doSgdStep)

        return pure(log, PX3[DL, InputOutput, ENV]())

    model = eqx.filter_jit(traverse(updateStep()).func)

    start = time.time()
    logs, trained_env = model(dialect, dataloader, initEnv)
    tttt = f"Train Time: {time.time() - start}"

    # logs = logs.value
    # print(logs.train_loss)
    # print(logs.test_loss)

    loss, _ = lossFn(dialect, dataloader, trained_env)
    print(loss / t_series_length)
    test_loss, _ = lossFn(dialect, test_set, trained_env)
    print(test_loss / test_steps)
    test_loss = test_loss / test_steps
    # test_loss_init, _ = lossFn(dialect, test_set, initEnv)
    # print(test_loss_init / test_steps)
    print(tttt)
    return logs


# def main2():
#     results = []

#     for t1 in range(20):
#         t2 = t1 + 2
#         N = 1_000_000
#         N_test = 10_000
#         rng_key = jax.random.key(54)
#         rng_key, prng1, _ = jax.random.split(rng_key, 3)

#         X, Y = generate_add_task_dataset(N, t1, t2, 1, prng1)
#         X_test, Y_test = generate_add_task_dataset(N_test, t1, t2, 1, rng_key)

#         dataloader = Traversable(InputOutput(x=X, y=Y))
#         test_set = Traversable(InputOutput(x=X_test, y=Y_test))

#         test_loss = onlineLearnerLoop(dataloader, N, test_set, N_test)

#         results.append(test_loss)  # Collect test losses

#     results_array = jnp.array(results)  # Convert list to JAX array

#     # Save results using JAX's save function
#     jax.numpy.save("rflo.npy", results_array)
#     print("Results saved to rflo.npy")


def main2():
    t1 = 5
    t2 = 9
    N = 3_000
    N_test = 10_000
    rng_key = jax.random.key(54)
    rng_key, prng1, _ = jax.random.split(rng_key, 3)

    X, Y = generate_add_task_dataset(N, t1, t2, 1, prng1)
    X_test, Y_test = generate_add_task_dataset(N_test, t1, t2, 1, rng_key)

    dataloader = Traversable(InputOutput(x=X, y=Y))
    test_set = Traversable(InputOutput(x=X_test, y=Y_test))

    logs = onlineLearnerLoop(dataloader, N, test_set, N_test, 0.01)

    # Save results using JAX's save function
    # jax.numpy.save("uoro_train.npy", logs.value.train_loss)
    # print("Results saved")


def main():
    t1 = 5
    t2 = 9
    N = 30_000
    N_val = 2_000
    N_test = 10_000
    t_series_length = 10  # how much time series goines into ONE param update
    trunc_length = 1  # controls how much avging done in one t_series
    # if trunc_length = 1, then divide by t_series_length. if trunc_length = t_series_length, then no normalization done
    rng_key = jax.random.key(54)
    rng_key, prng1, prng2 = jax.random.split(rng_key, 3)
    X, Y = generate_add_task_dataset(N, t1, t2, 1, prng1)
    X_val, Y_val = generate_add_task_dataset(N_val, t1, t2, 1, prng2)
    X_test, Y_test = generate_add_task_dataset(N_test, t1, t2, 1, rng_key)

    def transform(arr: Array, _t: int):
        return arr.reshape((_t, -1) + arr.shape[1:])

    numUpdates = N // t_series_length

    X = transform(X, numUpdates)
    Y = transform(Y, numUpdates)
    # for every param update, go through whole validation
    X_val = jnp.repeat(jnp.expand_dims(X_val, axis=0), numUpdates, axis=0)
    Y_val = jnp.repeat(jnp.expand_dims(Y_val, axis=0), numUpdates, axis=0)

    dataloader = Traversable(
        OhoInputOutput(
            train=Traversable(InputOutput(x=X, y=Y)),
            validation=Traversable(InputOutput(x=X_val, y=Y_val)),
            labels=Traversable(Y_val),
        )
    )

    test_set = Traversable(InputOutput(x=X_test, y=Y_test))

    logs, test_loss = ohoLoop(dataloader, t_series_length, trunc_length, N, test_set, N_test, 0.01)

    learning_rates = logs.learning_rate

    # Plot the arrays
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, label="learning_rates", color="blue")
    plt.legend()
    plt.xlabel("Update Steps")
    plt.ylabel("Learning Rate")

    plt.savefig("learning_rates.png", dpi=300)  # You can change the dpi for quality
    plt.close()


main2()

# import matplotlib

# matplotlib.use("Agg")


# def moving_average(arr, window_size):
#     """Compute the moving average with a given window size."""
#     return jnp.convolve(arr, jnp.ones(window_size) / window_size, mode="valid")[::window_size]


# window_size = 1000
# y1 = jnp.load("rtrl_train.npy")
# y1 = moving_average(y1, window_size)
# y2 = jnp.load("uoro_train.npy")
# y2 = moving_average(y2, window_size)
# y3 = jnp.load("rflo_train.npy")
# y3 = moving_average(y3, window_size)

# # Plot the arrays
# plt.figure(figsize=(10, 6))
# plt.plot(y1, label="RTRL", color="blue")
# plt.plot(y2, label="UORO", color="green")
# plt.plot(y3, label="RFLO", color="pink")
# plt.legend()
# plt.xlabel("Update Steps")
# plt.ylabel("Train Loss")
# plt.ylim(0.45, 0.6)
# # plt.xticks(range(len(y1)))

# plt.savefig("learning_train.png", dpi=300)  # You can change the dpi for quality
# plt.close()
