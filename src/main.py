# %%
import argparse
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
1) implement vanilla rnn training loop. Done.
2) implement oho to show how easy it is. Done.
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
    meta_learning_rate = jnp.asarray([0.00])

    rng_key, subkey1, subkey2, subkey3 = jax.random.split(rng_key, 4)

    W_in = jax.random.normal(subkey1, (n_h, n_in)) * jnp.sqrt(1 / n_in)
    W_rec, _ = jnp.linalg.qr(jax.random.normal(subkey2, (n_h, n_h)))  # QR decomposition
    W_out = jax.random.normal(subkey3, (n_out, n_h)) * jnp.sqrt(1 / n_h)
    b_rec = jnp.zeros((n_h, 1))
    b_out = jnp.zeros((n_out, 1))

    w_rec = jnp.hstack([W_rec, W_in, b_rec])
    w_out = jnp.hstack([W_out, b_out])

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
    num_activ = jnp.size(activation)
    num_params = jnp.size(compose2(endowVector, toVector)(parameter))
    influenceTensor_ = zeroedInfluenceTensor(n_h, parameter)
    ohoInfluenceTensor_ = zeroedInfluenceTensor(jnp.size(compose2(endowVector, toVector)(parameter)), sgd)
    init_env = RnnGodState[RnnParameter, SgdParameter, SgdParameter](
        activation=activation,
        influenceTensor=Jacobian[RnnParameter](influenceTensor_),
        ohoInfluenceTensor=Jacobian[SgdParameter](ohoInfluenceTensor_),
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
        logs=Logs(
            gradient=jnp.zeros_like(toVector(endowVector(parameter))),
            influenceTensor=influenceTensor_,
            immediateInfluenceTensor=influenceTensor_,
            hessian=jnp.zeros((num_activ, num_activ)),
        ),
        oho_logs=Logs(
            gradient=jnp.zeros_like(toVector(endowVector(sgd))),
            validationGradient=jnp.zeros_like(toVector(endowVector(parameter))),
            influenceTensor=ohoInfluenceTensor_,
            immediateInfluenceTensor=ohoInfluenceTensor_,
            hessian=jnp.zeros((num_params, num_params)),
        ),
    )

    return rng_key, init_env


def train(
    dataloader: Traversable[OhoInputOutput],
    test_set: Traversable[InputOutput],
    t_series_length: int,
    trunc_length: int,
    N_test: int,
    N_val: int,
    initEnv: RnnGodState[RnnParameter, SgdParameter, SgdParameter],
):
    type Train_Dl = BaseRnnInterpreter[RnnParameter, SgdParameter, SgdParameter]
    type OHO = OhoInterpreter[RnnParameter, SgdParameter, SgdParameter]
    type ENV = RnnGodState[RnnParameter, SgdParameter, SgdParameter]

    # interpreters
    trainDialect = BaseRnnInterpreter[RnnParameter, SgdParameter, SgdParameter]()
    ohoDialect = OhoInterpreter[RnnParameter, SgdParameter, SgdParameter](trainDialect)

    # lambda key, shape: jax.random.uniform(key, shape, minval=-1.0, maxval=1.0)
    onlineLearner: RnnLibrary[Train_Dl, InputOutput, ENV, PREDICTION, RnnParameter]
    onlineLearner = RTRL[
        InputOutput,
        ENV,
        ACTIVATION,
        RnnParameter,
        jax.Array,
        PREDICTION,
    ]().createLearner(
        doRnnStep(),
        doRnnReadout(),
        lambda a, b: LOSS(optax.safe_softmax_cross_entropy(a, b)),
        readoutRecurrentError(doRnnReadout(), lambda a, b: LOSS(optax.safe_softmax_cross_entropy(a, b))),
    )

    onlineLearner_folded = foldrRnnLearner(onlineLearner, initEnv.parameter)
    rnnLearner = endowAveragedGradients(onlineLearner_folded, trunc_length, initEnv.parameter)
    rnnLearner = logGradient(rnnLearner)
    # rnnLearner = normalizeGradientRnnLibrary(rnnLearner)

    oho_rtrl = RTRL[
        OhoInputOutput,
        ENV,
        RnnParameter,
        SgdParameter,
        jax.Array,
        Traversable[PREDICTION],
    ]()
    # oho_rtrl = IdentityLearner()

    def validation_loss(label: Traversable[jax.Array], prediction: Traversable[PREDICTION]):
        return LOSS(jnp.mean(optax.safe_softmax_cross_entropy(label.value, prediction.value)))

    oho: RnnLibrary[OHO, OhoInputOutput, ENV, Traversable[PREDICTION], SgdParameter]
    oho = endowBilevelOptimization(
        rnnLearner,
        doSgdStep,
        trainDialect,
        oho_rtrl,
        validation_loss,
        resetRnnActivation(initEnv.activation),
    )
    oho = logGradient(oho)
    # oho = clipGradient(4.0, oho)

    @do()
    def updateStep():
        env = yield from get(PX[ENV]())
        interpreter = yield from askForInterpreter(PX[OHO]())
        oho_data = yield from ask(PX[OhoInputOutput]())
        validation_model = resetRnnActivation(initEnv.activation).then(rnnLearner.rnnWithLoss).func

        learning_rate = yield from interpreter.getParameter().fmap(lambda x: x.learning_rate)
        te, _ = validation_model(trainDialect, test_set, env)
        vl, _ = validation_model(trainDialect, oho_data.validation, env)  # lag 1 timestep bc show prev param validation
        tr, _ = rnnLearner.rnnWithLoss.func(trainDialect, oho_data.train, env)
        weights = toVector(endowVector(env.parameter))

        _ = yield from oho.rnnWithGradient.flat_map(doSgdStep_Positive)

        log = AllLogs(
            trainLoss=tr / t_series_length,
            validationLoss=vl / N_val,
            testLoss=te / N_test,
            learningRate=learning_rate,
            parameterNorm=jnp.linalg.norm(weights),
            ohoGradient=env.oho_logs.gradient,
            trainGradient=jnp.linalg.norm(env.logs.gradient),
            validationGradient=jnp.linalg.norm(env.oho_logs.validationGradient),
            immediateInfluenceTensor=jnp.linalg.norm(jnp.ravel(env.oho_logs.immediateInfluenceTensor)),
            influenceTensor=jnp.linalg.norm(jnp.ravel(env.oho_logs.influenceTensor)),
            hessian=jnp.linalg.eigvals(env.oho_logs.hessian),
        )
        return pure(log, PX3[OHO, OhoInputOutput, ENV]())

    model = eqx.filter_jit(traverse(updateStep()).func).lower(ohoDialect, dataloader, initEnv).compile()

    start = time.time()
    logs, trained_env = model(ohoDialect, dataloader, initEnv)
    tttt = f"Train Time: {time.time() - start}"
    logs: AllLogs = logs.value

    print(logs.trainLoss[-1])
    print(logs.testLoss[-1])
    print(logs.validationLoss[-1])
    print(logs.learningRate[-1])
    print(tttt)

    return logs, trained_env


t1 = 15
t2 = 17
N = 100_000
N_val = 2000
N_test = 5000
t_series_length = 100  # how much time series goines into ONE param update
trunc_length = 1  # controls how much avging done in one t_series
# if trunc_length = 1, then divide by t_series_length. if trunc_length = t_series_length, then no normalization done
rng_key = jax.random.key(3241234)  # 54, 333, 3241234, 2342
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

rng_key, prng = jax.random.split(rng_key)
rng_key, initEnv = constructRnnEnv(prng, 0.1)

logs, test_loss = train(dataloader, test_set, t_series_length, trunc_length, N_test, N_val, initEnv)

filename = f"src/mytest55"

eqx.tree_serialise_leaves(f"{filename}.eqx", logs)

# Create the figure and subplots
fig, (ax1, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(
    8, 1, figsize=(12, 20), gridspec_kw={"height_ratios": [2, 1, 1, 1, 1, 1, 1, 1]}
)

# First subplot (Losses and Learning Rate)
ax1.plot(logs.trainLoss, label="Train Loss", color="blue")
ax1.plot(logs.validationLoss, label="Validation Loss", color="red")
ax1.plot(logs.testLoss, label="Test Loss", color="green")

ax2 = ax1.twinx()
ax2.plot(logs.learningRate, label="Learning Rate", color="purple", linestyle="dashed")

# Labels and legends for the first subplot
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax2.set_ylabel("Learning Rate")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
ax1.set_title("Training Progress")

# Second subplot (Parameter Norm)
ax3.plot(logs.parameterNorm, label="Parameter Norm", color="orange")
ax3.set_xlabel("Epochs")
ax3.set_ylabel("Parameter Norm")
ax3.legend(loc="upper right")
ax3.set_title("Parameter Norm Over Time")

# Third subplot (Oho Gradient)
ax4.plot(logs.ohoGradient, label="Oho Gradient", color="cyan")
ax4.set_xlabel("Epochs")
ax4.set_ylabel("Oho Gradient")
ax4.legend(loc="upper right")
ax4.set_title("Oho Gradient Over Time")

# Fourth subplot (Train Gradient)
ax5.plot(logs.trainGradient, label="Train Gradient", color="magenta")
ax5.set_xlabel("Epochs")
ax5.set_ylabel("Train Gradient")
ax5.legend(loc="upper right")
ax5.set_title("Train Gradient Over Time")

# Fifth subplot (Validation Gradient)
ax6.plot(logs.validationGradient, label="Validation Gradient", color="brown")
ax6.set_xlabel("Epochs")
ax6.set_ylabel("Validation Gradient")
ax6.legend(loc="upper right")
ax6.set_title("Validation Gradient Over Time")

# Sixth subplot (Immediate Influence Tensor)
ax7.plot(logs.immediateInfluenceTensor, label="Immediate Influence Tensor", color="teal")
ax7.set_xlabel("Epochs")
ax7.set_ylabel("Immediate Influence Tensor")
ax7.legend(loc="upper right")
ax7.set_title("Immediate Influence Tensor Over Time")

# Seventh subplot (Influence Tensor)
ax8.plot(logs.influenceTensor, label="Influence Tensor", color="darkblue")
ax8.set_xlabel("Epochs")
ax8.set_ylabel("Influence Tensor")
ax8.legend(loc="upper right")
ax8.set_title("Influence Tensor Over Time")

# Eighth subplot (Hessian)
ax9.plot(logs.hessian, label="Hessian", color="limegreen")
ax9.set_xlabel("Epochs")
ax9.set_ylabel("Hessian")
ax9.legend(loc="upper right")
ax9.set_title("Hessian Over Time")

# Adjust layout and save
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Keep the main title from overlapping
plt.savefig(f"{filename}.png", dpi=300)
plt.close()


# def parseIO():
#     parser = argparse.ArgumentParser(description="Parse configuration parameters for RnnConfig and Config.")

#     # later on I want to be able to get rid of this and have my model and dataset downloaded agnostically, at the cost of type safety
#     parser.add_argument("--seq", type=int, required=True, help="Sequence length that goes into one epoch")
#     parser.add_argument("--trunc", type=int, required=True, help="controls how much avging done in one t_series")
#     parser.add_argument("--t1", type=int, required=True, help="Parameter t1")
#     parser.add_argument("--t2", type=int, required=True, help="Parameter t2")
#     # Arguments for RnnConfig
#     parser.add_argument("--n_in", type=int, required=True, help="Number of input features for the RNN")
#     parser.add_argument("--n_h", type=int, required=True, help="Number of hidden units for the RNN")
#     parser.add_argument("--n_out", type=int, required=True, help="Number of output features for the RNN")
#     parser.add_argument("--numTr", type=int, required=True, help="Number of training samples")
#     parser.add_argument("--numVl", type=int, required=True, help="Number of validation samples")
#     parser.add_argument("--numTe", type=int, required=True, help="Number of testing samples")

#     parser.add_argument("--meta_learning_rate", type=float, required=True, help="Meta Learning rate")
#     parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate")
#     parser.add_argument("--alpha", type=float, required=True, help="Time Constant")
#     parser.add_argument("--bilevel-alpha", type=float, required=True, help="Time Constant")

#     parser.add_argument(
#         "--innerOptimizerFn", type=str, choices=["SGD"], required=True, help="Optimizer function for Inner"
#     )
#     parser.add_argument(
#         "--outerOptimizerFn",
#         type=str,
#         choices=["SGD", "SGD_Clipped", "Exp"],
#         required=True,
#         help="Optimizer function for Outer",
#     )
#     parser.add_argument("--lossFn", type=str, choices=["ce"], required=True, help="Loss function")
#     parser.add_argument("--seed", type=int, required=True, help="Seed")
#     parser.add_argument("--projectName", type=str, required=True, help="Wandb project name")
#     parser.add_argument(
#         "--performance_samples", type=int, required=True, help="Number of samples to visualize performance"
#     )
#     parser.add_argument(
#         "--activation_fn", type=str, choices=["relu", "tanh"], required=True, help="Activation function (relu or tanh)"
#     )
#     parser.add_argument(
#         "--outer_learner", type=str, choices=["identity", "rtrl", "uoro", "rflo"], required=True, help="Oho Learner"
#     )
#     parser.add_argument(
#         "--inner_learner", type=str, choices=["rtrl", "uoro", "rflo"], required=True, help="Inner Learner"
#     )

#     # set mlr to 0 if you want what oho would do without doing anything

#     args = parser.parse_args()

#     match args.lossFn:
#         case "ce":
#             loss_function = lambda a, b: LOSS(optax.safe_softmax_cross_entropy(a, b))
#         case _:
#             raise ValueError("Currently only mse is supported as a loss function")

#     # Determine optimizer function
#     match args.optimizerFn:
#         case "Adam":
#             optimizer_fn = torch.optim.Adam
#         case "SGD":
#             optimizer_fn = torch.optim.SGD
#         case _:
#             raise ValueError("Currently only Adam and SGD are supported as optimizer functions")

#     # Placeholder for task initialization
#     match args.task:
#         case "Sparse":
#             task = Sparse(args.outT)
#         case "Random":
#             match args.randomType:
#                 case "Uniform":
#                     randType = Uniform()
#                 case "Normal":
#                     randType = Normal()
#                 case _:
#                     raise ValueError("Invalid random type")
#             task = Random(randType)
#         case "Wave":
#             task = Wave()
#         case _:
#             raise ValueError("Invalid task type")

#     match args.logger:
#         case "wandb":
#             logger = WandbLogger()
#         case "prettyprint":
#             logger = PrettyPrintLogger()
#         case _:
#             raise ValueError("Invalid logger type")

#     match args.init_scheme:
#         case "ZeroInit":
#             scheme = ZeroInit()
#         case "RandomInit":
#             scheme = RandomInit()
#         case "StaticRandomInit":
#             scheme = StaticRandomInit()
#         case _:
#             raise ValueError("Invalid init type")

#     match args.activation_fn:
#         case "relu":
#             activation_fn = torch.relu
#         case "tanh":
#             activation_fn = torch.tanh
#         case _:
#             raise ValueError("Invalid activation function")

#     rnnConfig = RnnConfig(
#         n_in=args.n_in,
#         n_h=args.n_h,
#         n_out=args.n_out,
#         num_layers=args.num_layers,
#         scheme=scheme,
#         activation=activation_fn,
#     )

#     config = Config(
#         task=task,
#         seq=args.seq,
#         numTr=args.numTr,
#         numVl=args.numVl,
#         numTe=args.numTe,
#         batch_size_tr=args.batch_size_tr,
#         batch_size_vl=args.batch_size_vl,
#         batch_size_te=args.batch_size_te,
#         t1=args.t1,
#         t2=args.t2,
#         num_epochs=args.num_epochs,
#         learning_rate=args.learning_rate,
#         rnnConfig=rnnConfig,
#         criterion=loss_function,
#         optimizerFn=optimizer_fn,
#         modelArtifact=ArtifactConfig(
#             artifact=lambda name: wandb.Artifact(f"model_{name}", type="model"), path=lambda x: f"model_{x}.pt"
#         ),
#         datasetArtifact=ArtifactConfig(
#             artifact=lambda name: wandb.Artifact(f"dataset_{name}", type="dataset"), path=lambda x: f"dataset_{x}.pt"
#         ),
#         checkpointFrequency=args.checkpoint_freq,
#         projectName=args.projectName,
#         seed=args.seed,
#         performanceSamples=args.performance_samples,
#         logFrequency=args.log_freq,
#         l2_regularization=args.l2_regularization,
#         meta_learning_rate=args.meta_learning_rate,
#         is_oho=is_oho,
#         time_chunk_size=args.time_chunk_size,
#         rnnInitialActivation=getRNNInit(rnnConfig.scheme, rnnConfig.num_layers, rnnConfig.n_h),
#     )


#     return args, config, logger


# def log_modelIO(config: Config, logger: Logger, model: RNN, name: str):
#     logger.log2External(config.modelArtifact, lambda path: torch.save(model.state_dict(), path), name)


# def log_datasetIO(config: Config, logger: Logger, dataset: TensorDataset, name: str):
#     logger.log2External(config.datasetArtifact, lambda path: torch.save(dataset, path), name)


# def main():
#     args, config, logger = parseIO()
#     torch.manual_seed(config.seed)

#     logger.init(config.projectName, args)
#     model = RNN(config.rnnConfig, config.learning_rate, config.l2_regularization)  # Random IO
#     # logger.watchPytorch(model)
#     log_modelIO(config, logger, model, "init")

#     ts = torch.arange(0, config.seq)
#     dataGenerator = getRandomTask(config.task)

#     train_ds = getDatasetIO(dataGenerator, config.t1, config.t2, ts, config.numTr)
#     train_loader = getDataLoaderIO(train_ds, config.batch_size_tr)
#     test_dataset = getDatasetIO(dataGenerator, config.t1, config.t2, ts, config.numTe)
#     test_loader = getDataLoaderIO(test_dataset, config.batch_size_te)
#     valid_ds = getDatasetIO(dataGenerator, config.t1, config.t2, ts, config.numVl)
#     valid_loader = getDataLoaderIO(valid_ds, config.batch_size_vl)

#     log_datasetIO(config, logger, train_ds, "train")
#     log_datasetIO(config, logger, test_dataset, "test")
#     log_datasetIO(config, logger, valid_ds, "valid")

#     # dataset_artifact = wandb.use_artifact('wlp9800-new-york-university/mlr-test/dataset_ef1inln6:v0', type='dataset')
#     # dataset_artifact_dir = dataset_artifact.download()
#     # dataset = torch.load(f"{dataset_artifact_dir}/dataset_train.pt")

#     # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [config.numTr, config.numVl])
#     # train_loader = getDataLoaderIO(train_dataset, config.batch_size_tr)
#     # valid_loader = getDataLoaderIO(val_dataset, config.batch_size_vl)

#     # test_dataset_artifact = wandb.use_artifact('wlp9800-new-york-university/mlr-test/dataset_ef1inln6:v1', type='dataset')
#     # test_dataset_artifact_dir = test_dataset_artifact.download()
#     # test_dataset = torch.load(f"{test_dataset_artifact_dir}/dataset_test.pt")
#     # test_loader = getDataLoaderIO(test_dataset, config.batch_size_te)

#     model = train(config, logger, model, train_loader, cycle(valid_loader), test_loader, test_dataset)


# if __name__ == "__main__":
#     main()
