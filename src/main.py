import pickle
from typing import Any

import jax.experimental
import optax
from recurrent.mylearning import *
from recurrent.mylearning import RFLO, RTRL, UORO, IdentityLearner
from recurrent.myrecords import GodConfig, GodInterpreter
from recurrent.mytypes import *


from matplotlib import pyplot as plt
from recurrent.parameters import (
    RnnConfig,
    RnnParameter,
    SgdParameter,
    UORO_Param,
)

from recurrent.util import *
import jax.numpy as jnp
import equinox as eqx
import jax
import wandb

# jax.config.update("jax_enable_x64", True)


def main():
    # Initialize a new wandb run
    with wandb.init(mode="offline") as run:
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        # config: Config = wandb.config
        config = GodConfig(**wandb.config)
        prng = PRNG(jax.random.key(config.seed))
        test_prng = PRNG(jax.random.key(config.test_seed))
        env_prng, data_prng = jax.random.split(prng, 2)

        lossFn = getLossFn(config)
        env, innerInterpreter, outerInterpreter = create_env(config, env_prng)
        oho_set, test_set = create_datasets(config, data_prng, test_prng)

        all_logs, trained_env = train(oho_set, test_set, lossFn, env, innerInterpreter, outerInterpreter, config)

        logs = all_logs.value
        for i in range(logs.trainLoss.shape[0]):
            run.log(
                {
                    "train_loss": logs.trainLoss[i],
                    "validation_loss": logs.validationLoss[i],
                    "test_loss": logs.testLoss[i],
                    "parameter_norm": logs.parameterNorm[i],
                    "oho_gradient_norm": logs.ohoGradient[i],
                    "train_gradient_norm": logs.trainGradient[i],
                    "validation_gradient_norm": logs.validationGradient[i],
                    "immediate_influence_tensor_norm": logs.immediateInfluenceTensorNorm[i],
                    "influence_tensor_norm": logs.influenceTensorNorm[i],
                    "hessian_eigenvalues": logs.hessian[i],
                }
            )

        trained_env_path = "trained_env.pkl"
        with open(trained_env_path, "wb") as f:
            pickle.dump(trained_env, f)

        # **Generate a unique artifact name using the W&B run ID**
        artifact_name = f"trained_env_{run.id}"

        artifact = wandb.Artifact(name=artifact_name, type="environment")
        artifact.add_file(trained_env_path)
        run.log_artifact(artifact)

        run.finish()


def getLossFn(config: GodConfig) -> Callable[[jax.Array, jax.Array], LOSS]:
    match config.lossFn:
        case "cross_entropy":
            return lambda a, b: LOSS(optax.safe_softmax_cross_entropy(a, b))
        case _:
            raise ValueError("Invalid loss function")


def create_env(config: GodConfig, prng: PRNG) -> tuple[GodState, GodInterpreter, GodInterpreter]:
    # These don't care how env is created, just tags all possible parameter/states and vectorizes them
    inner_states = [lambda s: s.rnnState.activation]
    inner_params = [lambda s: s.rnnState.rnnParameter]
    outer_states = [lambda s: s.rnnState.rnnParameter, lambda s: s.innerOptState]
    outer_params = [lambda s: s.innerSgdParameter, lambda s: s.innerAdamParameter]

    def extract_attrbs(godState: GodState, attrbs: list[Callable[[GodState], Any]]) -> list[Any]:
        return [attrb(godState) for attrb in attrbs if attrb(godState) is not None]

    def toArray(godState: GodState, attrbs: list[Callable[[GodState], Any]]) -> Array:
        return toVector(endowVector(extract_attrbs(godState, attrbs)))

    def fromArray(godState: GodState, attrbs: list[Callable[[GodState], Any]], value: Array) -> GodState:
        xs = invmap(extract_attrbs(godState, attrbs), lambda _: value)
        for x, f in zip(xs, attrbs):
            godState = eqx.tree_at(f, godState, x, is_leaf=lambda x: x is None)
        return godState

    inner_state_get, inner_state_set = lambda g: toArray(g, inner_states), lambda g, v: fromArray(g, inner_states, v)
    inner_param_get, inner_param_set = lambda g: toArray(g, inner_params), lambda g, v: fromArray(g, inner_params, v)
    outer_state_get, outer_state_set = lambda g: toArray(g, outer_states), lambda g, v: fromArray(g, outer_states, v)
    outer_param_get, outer_param_set = lambda g: toArray(g, outer_params), lambda g, v: fromArray(g, outer_params, v)

    def putter[T](env: GodState, f: Callable[[GodState], T], value: T) -> GodState:
        return eqx.tree_at(f, env, value, is_leaf=lambda x: x is None)

    prng1, prng2, prng3, prng4, prng5, prng6, prng7, prng8, prng9 = jax.random.split(prng, 9)
    env = GodState(
        prng=prng1,
        logConfig=LogConfig(doLog=config.logFlag),
        innerTimeConstant=config.inner_time_constant,
        outerTimeConstant=config.outer_time_constant,
    )

    # 1) Initialize inner state and parameters
    match config.activation_fn:
        case "tanh":
            activation_fn = jax.nn.tanh
        case "relu":
            activation_fn = jax.nn.relu
        case _:
            raise ValueError("Invalid activation function")

    match config.architecture:
        case "rnn":
            W_in = jax.random.normal(prng2, (config.n_h, config.n_in)) * jnp.sqrt(1 / config.n_in)
            W_rec, _ = jnp.linalg.qr(jax.random.normal(prng3, (config.n_h, config.n_h)))
            W_out = jax.random.normal(prng4, (config.n_out, config.n_h)) * jnp.sqrt(1 / config.n_h)
            b_rec = jnp.zeros((config.n_h, 1))
            b_out = jnp.zeros((config.n_out, 1))
            w_rec = jnp.hstack([W_rec, W_in, b_rec])
            w_out = jnp.hstack([W_out, b_out])
            rnn_parameter = RnnParameter(w_rec=w_rec, w_out=w_out)
            rnn_config = RnnConfig(n_h=config.n_h, n_in=config.n_in, n_out=config.n_out, activationFn=activation_fn)
            activation = ACTIVATION(jax.random.normal(prng5, (config.n_h,)))
            rnn_state = RnnState(rnnConfig=rnn_config, activation=activation, rnnParameter=rnn_parameter)

            env = putter(env, lambda s: s.rnnState, rnn_state)

            if config.inner_learner == "uoro":
                a_init = jax.random.normal(prng6, (rnn_config.n_h,))
                b_init = toVector(
                    endowVector(
                        RnnParameter(
                            w_rec=jax.random.normal(prng7, rnn_parameter.w_rec.shape),
                            w_out=jnp.zeros_like(rnn_parameter.w_out),
                        )
                    )
                )
                env = putter(env, lambda s: s.innerUoro, UORO_Param(A=a_init, B=b_init))

        case _:
            raise ValueError("Invalid architecture")

    rec_state_n = jnp.size(inner_state_get(env))
    rec_param_n = jnp.size(inner_param_get(env))

    # ============================
    # 2) Initialize inner optimizer

    match config.inner_optimizer:
        case "sgd" | "sgd_positive":
            sgd = SgdParameter(learning_rate=config.inner_learning_rate)
            opt_state = optax.sgd(config.inner_learning_rate).init(jnp.empty((rec_param_n,)))
            env = putter(env, lambda s: s.innerSgdParameter, sgd)
            env = putter(env, lambda s: s.innerOptState, opt_state)
            get_inner_optimizer = lambda s: optax.sgd(s.innerSgdParameter.learning_rate)
        case "adam":
            adam = AdamParameter(learning_rate=config.inner_learning_rate)
            opt_state = optax.adam(config.inner_learning_rate).init(jnp.empty((rec_param_n,)))
            env = putter(env, lambda s: s.innerAdamParameter, adam)
            env = putter(env, lambda s: s.innerOptState, opt_state)
            get_inner_optimizer = lambda s: optax.adam(s.innerAdamParameter.learning_rate)
        case _:
            raise ValueError("Invalid inner optimizer")

    match config.inner_optimizer:
        case "sgd_positive":

            def inner_updater(params: optax.Params, updates: optax.Updates):
                return jax.tree.map(lambda p, u: jnp.maximum(p + u, 0), params, updates)
        case _:
            inner_updater = optax.apply_updates

    inner_influence_tensor = JACOBIAN(jnp.zeros((rec_state_n, rec_param_n)))
    env = putter(env, lambda s: s.innerInfluenceTensor, inner_influence_tensor)

    innerLogs = Logs(
        gradient=jnp.zeros((rec_param_n,)),
        influenceTensor=inner_influence_tensor,
        immediateInfluenceTensor=inner_influence_tensor,
        hessian=jnp.zeros((rec_state_n, rec_state_n)),
    )
    env = putter(env, lambda s: s.innerLogs, innerLogs)

    # =============================
    # 3) Initialize outer state and parameters

    outer_rec_state_n = jnp.size(outer_state_get(env))
    outer_rec_param_n = jnp.size(outer_param_get(env))

    match config.outer_optimizer:
        case "sgd" | "sgd_positive":
            sgd = SgdParameter(learning_rate=config.outer_learning_rate)
            opt_state = optax.sgd(config.outer_learning_rate).init(jnp.empty((outer_rec_param_n,)))
            env = putter(env, lambda s: s.outerSgdParameter, sgd)
            env = putter(env, lambda s: s.outerOptState, opt_state)
            get_outer_optimizer = lambda s: optax.sgd(s.outerSgdParameter.learning_rate)
        case "adam":
            adam = AdamParameter(learning_rate=config.outer_learning_rate)
            opt_state = optax.adam(config.outer_learning_rate).init(jnp.empty((outer_rec_param_n,)))
            env = putter(env, lambda s: s.outerAdamParameter, adam)
            env = putter(env, lambda s: s.outerOptState, opt_state)
            get_outer_optimizer = lambda s: optax.adam(s.outerAdamParameter.learning_rate)
        case _:
            raise ValueError("Invalid outer optimizer")

    match config.outer_optimizer:
        case "sgd_positive":

            def outer_updater(params: optax.Params, updates: optax.Updates):
                return jax.tree.map(lambda p, u: jnp.maximum(p + u, 0), params, updates)
        case _:
            outer_updater = optax.apply_updates

    outer_influence_tensor = JACOBIAN(jnp.zeros((outer_rec_state_n, outer_rec_param_n)))
    env = putter(env, lambda s: s.outerInfluenceTensor, outer_influence_tensor)

    outerLogs = Logs(
        gradient=jnp.zeros((outer_rec_param_n,)),
        influenceTensor=outer_influence_tensor,
        immediateInfluenceTensor=outer_influence_tensor,
        hessian=jnp.zeros((outer_rec_state_n, outer_rec_state_n)),
    )
    env = putter(env, lambda s: s.outerLogs, outerLogs)

    match config.outer_learner:
        case "uoro":
            uoro = UORO_Param(
                A=jax.random.normal(prng8, (outer_rec_state_n,)),
                B=jax.random.normal(prng9, (outer_rec_state_n * outer_rec_param_n,)),
            )
            env = putter(env, lambda s: s.outerUoro, uoro)
        case _:
            pass

    # =============================
    def monadic_getter[T](f: Callable[[GodState], T]) -> App[GodInterpreter, GodState, T]:
        return get(PX[GodState]()).fmap(f)

    def monadic_putter[T](f: Callable[[GodState], T]) -> Callable[[T], App[GodInterpreter, GodState, Unit]]:
        return lambda x: modifies(lambda s: eqx.tree_at(f, s, x, is_leaf=lambda x: x is None))

    # 4) Create interpreters
    innerInterpreter = GodInterpreter(
        getReccurentState=monadic_getter(inner_state_get),
        putReccurentState=lambda x: modifies(lambda s: inner_state_set(s, x)),
        getRecurrentParam=monadic_getter(inner_param_get),
        putRecurrentParam=lambda x: modifies(lambda s: inner_param_set(s, x)),
        getActivation=monadic_getter(lambda s: s.rnnState.activation),
        putActivation=monadic_putter(lambda s: s.rnnState.activation),
        getInfluenceTensor=monadic_getter(lambda s: s.innerInfluenceTensor),
        putInfluenceTensor=monadic_putter(lambda s: s.innerInfluenceTensor),
        getUoro=monadic_getter(lambda s: s.innerUoro),
        putUoro=monadic_putter(lambda s: s.innerUoro),
        getRnnConfig=monadic_getter(lambda s: s.rnnState.rnnConfig),
        getTimeConstant=monadic_getter(lambda s: s.innerTimeConstant),
        getLogConfig=monadic_getter(lambda s: s.logConfig),
        putLogs=monadic_putter(lambda s: s.innerLogs),
        getRnnParameter=monadic_getter(lambda s: s.rnnState.rnnParameter),
        putRnnParameter=monadic_putter(lambda s: s.rnnState.rnnParameter),
        getOptState=monadic_getter(lambda s: s.innerOptState),
        putOptState=monadic_putter(lambda s: s.innerOptState),
        getOptimizer=monadic_getter(get_inner_optimizer),
        getUpdater=pure(inner_updater, PX[tuple[GodInterpreter, GodState]]()),
    )

    outerInterpreter = GodInterpreter(
        getReccurentState=monadic_getter(outer_state_get),
        putReccurentState=lambda x: modifies(lambda s: outer_state_set(s, x)),
        getRecurrentParam=monadic_getter(outer_param_get),
        putRecurrentParam=lambda x: modifies(lambda s: outer_param_set(s, x)),
        getActivation=pure(None, PX[tuple[GodInterpreter, GodState]]()),
        putActivation=lambda x: pure(None, PX[tuple[GodInterpreter, GodState]]()),
        getInfluenceTensor=monadic_getter(lambda s: s.outerInfluenceTensor),
        putInfluenceTensor=monadic_putter(lambda s: s.outerInfluenceTensor),
        getUoro=monadic_getter(lambda s: s.outerUoro),
        putUoro=monadic_putter(lambda s: s.outerUoro),
        getRnnConfig=pure(None, PX[tuple[GodInterpreter, GodState]]()),
        getTimeConstant=monadic_getter(lambda s: s.outerTimeConstant),
        getLogConfig=monadic_getter(lambda s: s.logConfig),
        putLogs=monadic_putter(lambda s: s.outerLogs),
        getRnnParameter=pure(None, PX[tuple[GodInterpreter, GodState]]()),
        putRnnParameter=lambda x: pure(None, PX[tuple[GodInterpreter, GodState]]()),
        getOptState=monadic_getter(lambda s: s.outerOptState),
        putOptState=monadic_putter(lambda s: s.outerOptState),
        getOptimizer=monadic_getter(get_outer_optimizer),
        getUpdater=pure(outer_updater, PX[tuple[GodInterpreter, GodState]]()),
    )

    return env, innerInterpreter, outerInterpreter


def create_datasets(
    config: GodConfig, prng: PRNG, test_rng_key: PRNG
) -> tuple[Traversable[OhoData[Traversable[InputOutput]]], Traversable[InputOutput]]:
    # Extract parameters from config
    t1, t2 = config.ts
    N_train = config.numTr
    N_val = config.numVal
    N_test = config.numTe
    tr_series_length = config.tr_examples_per_epoch
    vl_series_length = config.vl_examples_per_epoch

    # Initialize PRNG keys
    prng1, prng2 = jax.random.split(prng, 2)

    # Generate raw datasets
    X_train, Y_train = generate_add_task_dataset(N_train, t1, t2, 1, prng1)
    X_val, Y_val = generate_add_task_dataset(N_val, t1, t2, 1, prng2)
    X_test, Y_test = generate_add_task_dataset(N_test, t1, t2, 1, test_rng_key)

    # Calculate number of updates
    num_updates_train = N_train // tr_series_length
    num_updates_val = N_val // vl_series_length

    # Transform training data
    X_train = transform(X_train, num_updates_train)
    Y_train = transform(Y_train, num_updates_train)

    # Transform validation data
    X_val = transform(X_val, num_updates_val)
    Y_val = transform(Y_val, num_updates_val)

    # Repeat validation data to match training epochs if needed
    if num_updates_val < num_updates_train:
        repeats_needed = num_updates_train // num_updates_val + 1  # subsequent :num_updates_train trims down to exact
        X_val = jnp.repeat(X_val, repeats_needed, axis=0)[:num_updates_train]
        Y_val = jnp.repeat(Y_val, repeats_needed, axis=0)[:num_updates_train]
    elif num_updates_val > num_updates_train:
        X_val = X_val[:num_updates_train]
        Y_val = Y_val[:num_updates_train]

    # Create Traversable objects
    train_set = Traversable(InputOutput(x=X_train, y=Y_train))
    val_set = Traversable(InputOutput(x=X_val, y=Y_val))
    test_set = Traversable(InputOutput(x=X_test, y=Y_test))

    oho_set = Traversable(OhoData(payload=train_set, validation=val_set))

    return oho_set, test_set


def transform(arr: Array, _t: int):
    return arr.reshape((_t, -1) + arr.shape[1:])


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


def create_learner(learner: str) -> RTRL | RFLO | UORO | IdentityLearner:
    match learner:
        case "rtrl":
            return RTRL()
        case "rflo":
            return RFLO()
        case "uoro":
            return UORO(lambda key, shape: jax.random.uniform(key, shape, minval=-1.0, maxval=1.0))
        case "identity":
            return IdentityLearner()
        case _:
            raise ValueError("Invalid learner")


def create_rnn_learner(learner: RTRL | RFLO | UORO | IdentityLearner, lossFn: Callable[[jax.Array, jax.Array], LOSS]):
    match learner:
        case _:
            library: Library[InputOutput, GodInterpreter, GodState, PREDICTION]
            library = learner.createLearner(
                lambda d: doRnnStep(d).then(ask(PX[GodInterpreter]())).flat_map(lambda i: i.getReccurentState),
                doRnnReadout,  # these can be passed in dynamically later
                lambda a, b: lossFn(a, b.y),
                readoutRecurrentError(doRnnReadout, lossFn),
            )
            return foldrLibrary(library)


def train(
    dataset: Traversable[OhoData[Traversable[InputOutput]]],
    test_dataset: Traversable[InputOutput],
    lossFn: Callable[[jax.Array, jax.Array], LOSS],
    initEnv: GodState,
    innerInterpreter: GodInterpreter,
    outerInterpreter: GodInterpreter,
    config: GodConfig,
) -> tuple[Traversable[AllLogs], GodState]:
    innerLearner = create_learner(config.inner_learner)
    innerLibrary = create_rnn_learner(innerLearner, lossFn)
    outerLearner = create_learner(config.outer_learner)

    innerController = endowAveragedGradients(innerLibrary.modelGradient, config.tr_avg_per)
    innerController = logGradient(innerController)
    innerLibrary = innerLibrary._replace(modelGradient=innerController)

    match outerLearner:
        case "bptt":
            raise NotImplementedError("BPTT is not implemented yet")
        case _:
            outerLibrary = endowBilevelOptimization(
                innerLibrary,
                doOptimizerStep,
                innerInterpreter,
                outerLearner,
                lambda a, b: LOSS(jnp.mean(lossFn(a.value, b.validation.value.y))),
                resetRnnActivation(initEnv.rnnState.activation),
            )

    outerController = logGradient(outerLibrary.modelGradient)
    outerLibrary = outerLibrary._replace(modelGradient=outerController)

    @do()
    def updateStep(oho_data: OhoData[Traversable[InputOutput]]):
        env = yield from get(PX[GodState]())
        interpreter = yield from ask(PX[GodInterpreter]())
        hyperparameters = yield from interpreter.getRecurrentParam
        weights, _ = interpreter.getRecurrentParam.func(innerInterpreter, env)

        validation_model = (
            lambda ds: resetRnnActivation(initEnv.rnnState.activation).then(innerLibrary.modelLossFn(ds)).func
        )

        te, _ = validation_model(test_dataset)(innerInterpreter, env)
        vl, _ = validation_model(oho_data.validation)(innerInterpreter, env)
        tr, _ = innerLibrary.modelLossFn(oho_data.payload)(innerInterpreter, env)

        _ = yield from outerLibrary.modelGradient(oho_data).flat_map(doOptimizerStep)

        log = AllLogs(
            trainLoss=tr / config.tr_examples_per_epoch,
            validationLoss=vl / config.vl_examples_per_epoch,
            testLoss=te / config.numTe,
            hyperparameters=hyperparameters,
            parameterNorm=jnp.linalg.norm(weights),
            ohoGradient=env.outerLogs.gradient,
            trainGradient=env.innerLogs.gradient,
            validationGradient=env.outerLogs.validationGradient,
            immediateInfluenceTensorNorm=jnp.linalg.norm(jnp.ravel(env.outerLogs.immediateInfluenceTensor)),
            influenceTensorNorm=jnp.linalg.norm(jnp.ravel(env.outerLogs.influenceTensor)),
            hessian=jnp.linalg.eigvals(env.outerLogs.hessian),
        )
        return pure(log, PX[tuple[GodInterpreter, GodState]]())

    model = eqx.filter_jit(traverseM(updateStep)(dataset).func).lower(outerInterpreter, initEnv).compile()

    logs, trained_env = model(outerInterpreter, initEnv)
    return logs, trained_env


# # Test script
# def test_create_datasets():
#     # Create initialized GodConfig object
#     config = GodConfig(
#         inner_learning_rate=0.01,
#         outer_learning_rate=0.001,
#         ts=(1, 2),
#         seed=42,
#         test_seed=43,
#         tr_examples_per_epoch=10,
#         vl_examples_per_epoch=5,
#         tr_avg_per=1,
#         vl_avg_per=1,
#         num_epochs=100,
#         numVal=200,
#         numTr=1000,
#         numTe=100,
#         inner_learner="rtrl",
#         outer_learner="rtrl",
#         lossFn="cross_entropy",
#         inner_optimizer="sgd",
#         outer_optimizer="sgd",
#         activation_fn="tanh",
#         architecture="rnn",
#         n_h=10,
#         n_in=2,
#         n_out=2,
#         inner_time_constant=1.0,
#         outer_time_constant=1.0,
#         logFlag=False,
#     )

#     # Create PRNG keys
#     prng = jax.random.PRNGKey(config.seed)
#     test_prng = jax.random.PRNGKey(config.test_seed)

#     # Run your existing create_datasets function
#     dataset, test_set = create_datasets(config, prng, test_prng)

#     # Verify structure and shapes
#     print("Testing dataset structure and shapes:")

#     print("✓ Outer Traversable exists")
#     inner = dataset.value
#     print("✓ Inner Traversable exists")
#     oho_data = inner.value
#     print("✓ OhoData exists")

#     payload = oho_data.payload
#     print(f"Payload x shape: {payload.value.x.shape}")
#     print(f"Payload y shape: {payload.value.y.shape}")
#     assert payload.value.x.shape[0] == config.num_epochs
#     print("✓ Payload has correct epoch dimension")

#     validation = oho_data.validation
#     print(f"Validation x shape: {validation.value.x.shape}")
#     print(f"Validation y shape: {validation.value.y.shape}")
#     assert validation.value.x.shape[0] == config.num_epochs
#     print("✓ Validation has correct epoch dimension")

#     print(f"Test x shape: {test_set.value.x.shape}")
#     print(f"Test y shape: {test_set.value.y.shape}")
#     print("✓ Test set exists")

#     assert jnp.all(jnp.isfinite(payload.value.x))
#     assert jnp.all(jnp.isfinite(validation.value.x))
#     assert jnp.all(jnp.isfinite(test_set.value.x))
#     print("✓ All arrays contain finite values")


# if __name__ == "__main__":
#     test_create_datasets()
#     print("\nAll tests passed!")

# t1 = 15
# t2 = 17
# N = 100_000
# N_val = 2000
# N_test = 5000
# t_series_length = 100  # how much time series goines into ONE param update
# trunc_length = 1  # controls how much avging done in one t_series
# # if trunc_length = 1, then divide by t_series_length. if trunc_length = t_series_length, then no normalization done
# rng_key = jax.random.key(3241234)  # 54, 333, 3241234, 2342
# rng_key, prng1, prng2 = jax.random.split(rng_key, 3)
# X, Y = generate_add_task_dataset(N, t1, t2, 1, prng1)
# X_val, Y_val = generate_add_task_dataset(N_val, t1, t2, 1, prng2)
# X_test, Y_test = generate_add_task_dataset(N_test, t1, t2, 1, rng_key)


# def transform(arr: Array, _t: int):
#     return arr.reshape((_t, -1) + arr.shape[1:])


# numUpdates = N // t_series_length
# X = transform(X, numUpdates)
# Y = transform(Y, numUpdates)
# # for every param update, go through whole validation
# X_val = jnp.repeat(jnp.expand_dims(X_val, axis=0), numUpdates, axis=0)
# Y_val = jnp.repeat(jnp.expand_dims(Y_val, axis=0), numUpdates, axis=0)

# dataloader = Traversable(
#     OhoInputOutput(
#         train=Traversable(InputOutput(x=X, y=Y)),
#         validation=Traversable(InputOutput(x=X_val, y=Y_val)),
#         labels=Traversable(Y_val),
#     )
# )

# test_set = Traversable(InputOutput(x=X_test, y=Y_test))

# rng_key, prng = jax.random.split(rng_key)
# rng_key, initEnv = constructRnnEnv(prng, 0.1)

# logs, test_loss = train(dataloader, test_set, t_series_length, trunc_length, N_test, N_val, initEnv)

# filename = f"src/mytest55"

# eqx.tree_serialise_leaves(f"{filename}.eqx", logs)

# # Create the figure and subplots
# fig, (ax1, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(
#     8, 1, figsize=(12, 20), gridspec_kw={"height_ratios": [2, 1, 1, 1, 1, 1, 1, 1]}
# )

# # First subplot (Losses and Learning Rate)
# ax1.plot(logs.trainLoss, label="Train Loss", color="blue")
# ax1.plot(logs.validationLoss, label="Validation Loss", color="red")
# ax1.plot(logs.testLoss, label="Test Loss", color="green")

# ax2 = ax1.twinx()
# ax2.plot(logs.learningRate, label="Learning Rate", color="purple", linestyle="dashed")

# # Labels and legends for the first subplot
# ax1.set_xlabel("Epochs")
# ax1.set_ylabel("Loss")
# ax2.set_ylabel("Learning Rate")
# ax1.legend(loc="upper left")
# ax2.legend(loc="upper right")
# ax1.set_title("Training Progress")

# # Second subplot (Parameter Norm)
# ax3.plot(logs.parameterNorm, label="Parameter Norm", color="orange")
# ax3.set_xlabel("Epochs")
# ax3.set_ylabel("Parameter Norm")
# ax3.legend(loc="upper right")
# ax3.set_title("Parameter Norm Over Time")

# # Third subplot (Oho Gradient)
# ax4.plot(logs.ohoGradient, label="Oho Gradient", color="cyan")
# ax4.set_xlabel("Epochs")
# ax4.set_ylabel("Oho Gradient")
# ax4.legend(loc="upper right")
# ax4.set_title("Oho Gradient Over Time")

# # Fourth subplot (Train Gradient)
# ax5.plot(logs.trainGradient, label="Train Gradient", color="magenta")
# ax5.set_xlabel("Epochs")
# ax5.set_ylabel("Train Gradient")
# ax5.legend(loc="upper right")
# ax5.set_title("Train Gradient Over Time")

# # Fifth subplot (Validation Gradient)
# ax6.plot(logs.validationGradient, label="Validation Gradient", color="brown")
# ax6.set_xlabel("Epochs")
# ax6.set_ylabel("Validation Gradient")
# ax6.legend(loc="upper right")
# ax6.set_title("Validation Gradient Over Time")

# # Sixth subplot (Immediate Influence Tensor)
# ax7.plot(logs.immediateInfluenceTensor, label="Immediate Influence Tensor", color="teal")
# ax7.set_xlabel("Epochs")
# ax7.set_ylabel("Immediate Influence Tensor")
# ax7.legend(loc="upper right")
# ax7.set_title("Immediate Influence Tensor Over Time")

# # Seventh subplot (Influence Tensor)
# ax8.plot(logs.influenceTensor, label="Influence Tensor", color="darkblue")
# ax8.set_xlabel("Epochs")
# ax8.set_ylabel("Influence Tensor")
# ax8.legend(loc="upper right")
# ax8.set_title("Influence Tensor Over Time")

# # Eighth subplot (Hessian)
# ax9.plot(logs.hessian, label="Hessian", color="limegreen")
# ax9.set_xlabel("Epochs")
# ax9.set_ylabel("Hessian")
# ax9.legend(loc="upper right")
# ax9.set_title("Hessian Over Time")

# # Adjust layout and save
# plt.tight_layout(rect=[0, 0, 1, 0.96])  # Keep the main title from overlapping
# plt.savefig(f"{filename}.png", dpi=300)
# plt.close()


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
#     parser.add_argument(
#         "--activation_fn", type=str, choices=["relu", "tanh"], required=True, help="Activation function (relu or tanh)"
#     )
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


# # def log_modelIO(config: Config, logger: Logger, model: RNN, name: str):
# #     logger.log2External(config.modelArtifact, lambda path: torch.save(model.state_dict(), path), name)


# # def log_datasetIO(config: Config, logger: Logger, dataset: TensorDataset, name: str):
# #     logger.log2External(config.datasetArtifact, lambda path: torch.save(dataset, path), name)


# # def main():
# #     args, config, logger = parseIO()
# #     torch.manual_seed(config.seed)

# #     logger.init(config.projectName, args)
# #     model = RNN(config.rnnConfig, config.learning_rate, config.l2_regularization)  # Random IO
# #     # logger.watchPytorch(model)
# #     log_modelIO(config, logger, model, "init")

# #     ts = torch.arange(0, config.seq)
# #     dataGenerator = getRandomTask(config.task)

# #     train_ds = getDatasetIO(dataGenerator, config.t1, config.t2, ts, config.numTr)
# #     train_loader = getDataLoaderIO(train_ds, config.batch_size_tr)
# #     test_dataset = getDatasetIO(dataGenerator, config.t1, config.t2, ts, config.numTe)
# #     test_loader = getDataLoaderIO(test_dataset, config.batch_size_te)
# #     valid_ds = getDatasetIO(dataGenerator, config.t1, config.t2, ts, config.numVl)
# #     valid_loader = getDataLoaderIO(valid_ds, config.batch_size_vl)

# #     log_datasetIO(config, logger, train_ds, "train")
# #     log_datasetIO(config, logger, test_dataset, "test")
# #     log_datasetIO(config, logger, valid_ds, "valid")

# #     # dataset_artifact = wandb.use_artifact('wlp9800-new-york-university/mlr-test/dataset_ef1inln6:v0', type='dataset')
# #     # dataset_artifact_dir = dataset_artifact.download()
# #     # dataset = torch.load(f"{dataset_artifact_dir}/dataset_train.pt")

# #     # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [config.numTr, config.numVl])
# #     # train_loader = getDataLoaderIO(train_dataset, config.batch_size_tr)
# #     # valid_loader = getDataLoaderIO(val_dataset, config.batch_size_vl)

# #     # test_dataset_artifact = wandb.use_artifact('wlp9800-new-york-university/mlr-test/dataset_ef1inln6:v1', type='dataset')
# #     # test_dataset_artifact_dir = test_dataset_artifact.download()
# #     # test_dataset = torch.load(f"{test_dataset_artifact_dir}/dataset_test.pt")
# #     # test_loader = getDataLoaderIO(test_dataset, config.batch_size_te)

# #     model = train(config, logger, model, train_loader, cycle(valid_loader), test_loader, test_dataset)


# # if __name__ == "__main__":
# #     main()
