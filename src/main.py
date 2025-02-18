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

jax.config.update("jax_enable_x64", True)

"""
Todo
1) implement vanilla rnn training loop 
2) implement oho to show how easy it is
3) implement feedforward to show how easy it is
"""


def generate_add_task_dataset(N, t_1, t_2, tau_task, rng_key):
    """y(t) = 0.5 + 0.5 * x(t - t_1) - 0.25 * x(t - t_2)"""
    N = N // tau_task

    x = jax.random.bernoulli(rng_key, 0.5, (N,)).astype(jnp.float64)

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
    meta_learning_rate = jnp.asarray([0.0005])

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
        logs=Logs(gradient=jnp.zeros_like(toVector(endowVector(parameter))), influenceTensor=influenceTensor_),
        oho_logs=Logs(
            gradient=jnp.zeros_like(toVector(endowVector(sgd))),
            validationGradient=jnp.zeros_like(toVector(endowVector(parameter))),
            influenceTensor=ohoInfluenceTensor_,
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
    initLearningRate: float,
):
    type Train_Dl = BaseRnnInterpreter[RnnParameter, SgdParameter, SgdParameter]
    type OHO = OhoInterpreter[RnnParameter, SgdParameter, SgdParameter]
    type ENV = RnnGodState[RnnParameter, SgdParameter, SgdParameter]

    # interpreters
    trainDialect = BaseRnnInterpreter[RnnParameter, SgdParameter, SgdParameter]()
    ohoDialect = OhoInterpreter[RnnParameter, SgdParameter, SgdParameter](trainDialect)

    # setup
    rng_key = jax.random.key(444)
    rng_key, prng = jax.random.split(rng_key)
    rng_key, initEnv = constructRnnEnv(prng, initLearningRate)

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
    oho_rtrl = IdentityLearner()

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

        _ = yield from oho.rnnWithGradient.flat_map(doSgdStep_Clipped)

        log = AllLogs(
            trainLoss=tr / t_series_length,
            validationLoss=vl / N_val,
            testLoss=te / N_test,
            learningRate=learning_rate,
            parameterNorm=weights,
            ohoGradient=env.oho_logs.gradient,
            trainGradient=env.logs.gradient,
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

# print(dataloader.value.train.value.x.shape)
# quit()

test_set = Traversable(InputOutput(x=X_test, y=Y_test))

logs, test_loss = train(dataloader, test_set, t_series_length, trunc_length, N_test, N_val, 0.3)

filename = f"src/mytest46"

eqx.tree_serialise_leaves(f"{filename}.eqx", logs)


fig, (ax1, ax3, ax4, ax5) = plt.subplots(4, 1, figsize=(12, 12), gridspec_kw={"height_ratios": [2, 1, 1, 1]})

# First subplot (Losses and Learning Rate)
ax1.plot(logs.trainLoss, label="Train Loss", color="blue")
ax1.plot(logs.validationLoss, label="Validation Loss", color="red")
ax1.plot(logs.testLoss, label="Test Loss", color="green")

ax2 = ax1.twinx()
ax2.plot(logs.learningRate, label="Learning Rate", color="purple", linestyle="dashed")

# Labels and legends
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

# Adjust layout and save
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Keep the main title from overlapping
plt.savefig(f"{filename}.png", dpi=300)
plt.close()


# learning_rates = logs.learning_rate

# # Plot the arrays
# plt.figure(figsize=(10, 6))
# plt.plot(learning_rates, label="learning_rates", color="blue")
# plt.legend()
# plt.xlabel("Update Steps")
# plt.ylabel("Learning Rate")

# plt.savefig("learning_rates.png", dpi=300)  # You can change the dpi for quality
# plt.close()


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


# class WandbLogger(Logger):
#     def log2External(self, artifactConfig: ArtifactConfig, saveFile: Callable[[str], None], name: str):
#         path = artifactConfig.path(name)
#         saveFile(path)
#         artifact = artifactConfig.artifact(wandb.run.id)
#         artifact.add_file(path)
#         wandb.log_artifact(artifact)

#     def log(self, dict: dict[str, Any]):
#         wandb.log(dict)

#     def init(self, projectName: str, config: argparse.Namespace):
#         wandb.login(key=os.getenv("WANDB_API_KEY"))
#         wandb.init(project=projectName, config=config)

#     def watchPytorch(self, model: nn.Module):
#         wandb.watch(model, log_freq=1000, log="all")


# class PrettyPrintLogger(Logger):
#     def log2External(self, artifactConfig: ArtifactConfig, saveFile: Callable[[str], None], name: str):
#         print(f"Saving {name} to {artifactConfig.path(name)}")
#         saveFile(artifactConfig.path(name))
#         print(f"Saved {name} to {artifactConfig.path(name)}")

#     def log(self, dict: dict[str, Any]):
#         for key, value in dict.items():
#             print(f"{key}: {value}")

#     def init(self, projectName: str, config: argparse.Namespace):
#         print(f"Initialized project {projectName} with config {config}")

#     def watchPytorch(self, model: nn.Module):
#         print("Model is being watched")


# def parseIO():
#     parser = argparse.ArgumentParser(description="Parse configuration parameters for RnnConfig and Config.")

#     # Arguments for RnnConfig
#     parser.add_argument("--n_in", type=int, required=True, help="Number of input features for the RNN")
#     parser.add_argument("--n_h", type=int, required=True, help="Number of hidden units for the RNN")
#     parser.add_argument("--n_out", type=int, required=True, help="Number of output features for the RNN")
#     parser.add_argument("--num_layers", type=int, required=True, help="Number of layers in the RNN")

#     # Arguments for Config
#     parser.add_argument(
#         "--task",
#         type=str,
#         choices=["Random", "Sparse", "Wave"],
#         required=True,
#         help="Task type (Random, Sparse, or Wave)",
#     )
#     parser.add_argument(
#         "--randomType", type=str, choices=["Uniform", "Normal"], required=False, help="Random type (Uniform or Normal)"
#     )
#     parser.add_argument(
#         "--init_scheme",
#         type=str,
#         choices=["ZeroInit", "RandomInit", "StaticRandomInit"],
#         required=True,
#         help="Rnn init scheme type (ZeroInit, or RandomInit)",
#     )
#     parser.add_argument("--outT", type=int, help="Output time step (required if task is Sparse)")
#     parser.add_argument("--seq", type=int, required=True, help="Sequence length")
#     parser.add_argument("--numTr", type=int, required=True, help="Number of training samples")
#     parser.add_argument("--numVl", type=int, required=True, help="Number of validation samples")
#     parser.add_argument("--numTe", type=int, required=True, help="Number of testing samples")
#     parser.add_argument("--batch_size_tr", type=int, required=True, help="Training batch size")
#     parser.add_argument("--batch_size_vl", type=int, required=True, help="Validation batch size")
#     parser.add_argument("--batch_size_te", type=int, required=True, help="Testing batch size")
#     parser.add_argument("--t1", type=int, required=True, help="Parameter t1")
#     parser.add_argument("--t2", type=int, required=True, help="Parameter t2")
#     parser.add_argument("--num_epochs", type=int, required=True, help="Number of training epochs")
#     parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate")
#     parser.add_argument(
#         "--optimizerFn", type=str, choices=["Adam", "SGD"], required=True, help="Optimizer function (Adam or SGD)"
#     )
#     parser.add_argument(
#         "--lossFn", type=str, choices=["mse"], required=True, help="Loss function (currently only mse is supported)"
#     )
#     parser.add_argument(
#         "--mode", type=str, choices=["test", "experiment"], required=True, help="Execution mode (test or experiment)"
#     )
#     parser.add_argument(
#         "--checkpoint_freq", type=int, required=True, help="Frequency of checkpoints during training (in epochs)"
#     )
#     parser.add_argument("--seed", type=int, required=True, help="Pytorch seed")
#     parser.add_argument("--projectName", type=str, required=True, help="Wandb project name")
#     parser.add_argument(
#         "--logger", type=str, choices=["wandb", "prettyprint"], required=True, help="Choice of logger to use"
#     )
#     parser.add_argument(
#         "--performance_samples", type=int, required=True, help="Number of samples to visualize performance"
#     )
#     parser.add_argument(
#         "--activation_fn", type=str, choices=["relu", "tanh"], required=True, help="Activation function (relu or tanh)"
#     )
#     parser.add_argument(
#         "--log_freq", type=int, required=True, help="Frequency of logging during training (in iterations)"
#     )
#     parser.add_argument("--l2_regularization", type=float, required=True, help="Learning rate")
#     parser.add_argument("--meta_learning_rate", type=float, required=True, help="Meta Learning rate")
#     parser.add_argument("--is_oho", type=int, required=True, help="Is OHO")
#     parser.add_argument("--time_chunk_size", type=int, required=True, help="Time chunk size")

#     args = parser.parse_args()

#     # Validate Sparse task requires outT
#     if args.task == "Sparse" and args.outT is None:
#         parser.error("--outT is required when --task is Sparse")

#     if args.task == "Random" and args.randomType is None:
#         parser.error("--randomType is required when --task is Random")

#     match args.is_oho:
#         case 0:
#             is_oho = False
#         case 1:
#             is_oho = True
#         case _:
#             raise ValueError("Invalid is_oho value")

#     match args.lossFn:
#         case "mse":
#             loss_function = lambda x, y: torch.functional.F.mse_loss(x, y, reduction="none").sum(dim=1).mean(dim=0)
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
