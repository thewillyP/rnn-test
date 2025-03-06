import copy
import os
import pickle
import time
from typing import Any

import jax.experimental
import optax
import psutil
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

        start = time.time()
        all_logs, trained_env = train(oho_set, test_set, lossFn, env, innerInterpreter, outerInterpreter, config)
        jax.block_until_ready(all_logs)
        end = time.time()
        print(f"Training time: {end - start} seconds")
        logs = all_logs.value

        def safe_real_array(value, i):
            if value is None:
                return None
            value = value[i]
            # Check if any element in the array is not finite
            if not jnp.all(jnp.isfinite(value)):
                return None
            return jnp.real(value)

        for i in range(logs.trainLoss.shape[0]):
            run.log(
                {
                    "train_loss": safe_real_array(logs.trainLoss, i),
                    "validation_loss": safe_real_array(logs.validationLoss, i),
                    "test_loss": safe_real_array(logs.testLoss, i),
                    "hyperparameters": safe_real_array(logs.hyperparameters, i),
                    "parameter_norm": safe_real_array(logs.parameterNorm, i),
                    "oho_gradient": safe_real_array(logs.ohoGradient, i),
                    "train_gradient": safe_real_array(logs.trainGradient, i),
                    "validation_gradient": safe_real_array(logs.validationGradient, i),
                    "immediate_influence_tensor_norm": safe_real_array(logs.immediateInfluenceTensorNorm, i),
                    "inner_influence_tensor_norm": safe_real_array(logs.innerInfluenceTensorNorm, i),
                    "outer_influence_tensor_norm": safe_real_array(logs.outerInfluenceTensorNorm, i),
                    "largest_jacobian_eigenvalue": safe_real_array(logs.largest_jacobian_eigenvalue, i),
                    "largest_influence_eigenvalue": safe_real_array(logs.largest_hessian_eigenvalue, i),
                }
            )

        trained_env = copy.replace(trained_env, prng=jax.random.key_data(trained_env.prng))
        trained_env_path = "trained_env.eqx"
        eqx.tree_serialise_leaves(trained_env_path, trained_env)

        # Generate a unique artifact name using the W&B run ID
        artifact_name = f"trained_env"

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
    outer_states = inner_params + [lambda s: s.innerOptState]  # VERY IMPORTANT COMES FIRST
    outer_params = [lambda s: s.innerSgdParameter, lambda s: s.innerAdamParameter]

    def toArray(godState: GodState, attrbs: list[Callable[[GodState], Any]]) -> Array:
        exact = [attrb(godState) for attrb in attrbs if attrb(godState) is not None]
        return toVector(endowVector(exact))

    def fromArray(godState: GodState, attrbs: list[Callable[[GodState], Any]], value: Array) -> GodState:
        fs = [attrb for attrb in attrbs if attrb(godState) is not None]
        ys = [attrb(godState) for attrb in fs]
        xs = invmap(ys, lambda _: value)
        for x, f in zip(xs, fs):
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
        logConfig=LogConfig(log_special=config.log_special, lanczos_iterations=config.lanczos_iterations),
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
            i_std = config.initialization_std
            W_in = jax.random.normal(prng2, (config.n_h, config.n_in)) * jnp.sqrt(1 / config.n_in) * i_std
            W_rec = jnp.linalg.qr(jax.random.normal(prng3, (config.n_h, config.n_h)))[0] * i_std
            W_out = jax.random.normal(prng4, (config.n_out, config.n_h)) * jnp.sqrt(1 / config.n_h) * i_std

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
        case "sgd_normalized":
            sgd = SgdParameter(learning_rate=config.inner_learning_rate)
            opt_state = normalized_sgd(config.inner_learning_rate).init(jnp.empty((rec_param_n,)))
            env = putter(env, lambda s: s.innerSgdParameter, sgd)
            env = putter(env, lambda s: s.innerOptState, opt_state)
            get_inner_optimizer = lambda s: normalized_sgd(s.innerSgdParameter.learning_rate)
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
        validationGradient=None,
        influenceTensor=inner_influence_tensor,
        immediateInfluenceTensor=inner_influence_tensor,
        jac_eigenvalue=0.0,
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
        case "sgd_normalized":
            sgd = SgdParameter(learning_rate=config.outer_learning_rate)
            opt_state = normalized_sgd(config.outer_learning_rate).init(jnp.empty((outer_rec_param_n,)))
            env = putter(env, lambda s: s.outerSgdParameter, sgd)
            env = putter(env, lambda s: s.outerOptState, opt_state)
            get_outer_optimizer = lambda s: normalized_sgd(s.outerSgdParameter.learning_rate)
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
        validationGradient=jnp.zeros((rec_param_n,)),
        influenceTensor=outer_influence_tensor,
        immediateInfluenceTensor=outer_influence_tensor,
        jac_eigenvalue=0.0,
    )
    env = putter(env, lambda s: s.outerLogs, outerLogs)

    match config.outer_learner:
        case "uoro":
            uoro = UORO_Param(
                A=jax.random.normal(prng8, (outer_rec_state_n,)),
                B=jax.random.normal(prng9, (outer_rec_param_n,)),
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
        getRecurrentState=monadic_getter(inner_state_get),
        putRecurrentState=lambda x: modifies(lambda s: inner_state_set(s, x)),
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
        # putLogs=monadic_putter(lambda s: s.innerLogs),
        putLogs=lambda s: modifies(lambda e: eqx.tree_at(lambda t: t.innerLogs, e, eqx.combine(s, e.innerLogs))),
        getRnnParameter=monadic_getter(lambda s: s.rnnState.rnnParameter),
        putRnnParameter=monadic_putter(lambda s: s.rnnState.rnnParameter),
        getOptState=monadic_getter(lambda s: s.innerOptState),
        putOptState=monadic_putter(lambda s: s.innerOptState),
        getOptimizer=monadic_getter(get_inner_optimizer),
        getUpdater=pure(inner_updater, PX[tuple[GodInterpreter, GodState]]()),
    )

    outerInterpreter = GodInterpreter(
        getRecurrentState=monadic_getter(outer_state_get),
        putRecurrentState=lambda x: modifies(lambda s: outer_state_set(s, x)),
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
        putLogs=lambda s: modifies(lambda e: eqx.tree_at(lambda t: t.outerLogs, e, eqx.combine(s, e.outerLogs))),
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
    tau_task = int(1 / config.inner_time_constant) if config.tau_task else 1

    # Initialize PRNG keys
    prng1, prng2 = jax.random.split(prng, 2)

    # Generate raw datasets
    X_train, Y_train = generate_add_task_dataset(N_train, t1, t2, tau_task, prng1)
    X_val, Y_val = generate_add_task_dataset(N_val, t1, t2, tau_task, prng2)
    X_test, Y_test = generate_add_task_dataset(N_test, t1, t2, tau_task, test_rng_key)

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


def create_learner(learner: str, rtrl_use_fwd: bool, uoro_std) -> RTRL | RFLO | UORO | IdentityLearner:
    match learner:
        case "rtrl":
            return RTRL(rtrl_use_fwd)
        case "rflo":
            return RFLO()
        case "uoro":
            return UORO(lambda key, shape: jax.random.uniform(key, shape, minval=-uoro_std, maxval=uoro_std))
        case "identity":
            return IdentityLearner()
        case _:
            raise ValueError("Invalid learner")


def create_rnn_learner(learner: RTRL | RFLO | UORO | IdentityLearner, lossFn: Callable[[jax.Array, jax.Array], LOSS]):
    lfn = lambda a, b: lossFn(a, b.y)
    match learner:
        case _:
            library: Library[InputOutput, GodInterpreter, GodState, PREDICTION]
            library = learner.createLearner(
                lambda d: doRnnStep(d).then(ask(PX[GodInterpreter]())).flat_map(lambda i: i.getRecurrentState),
                doRnnReadout,  # these can be passed in dynamically later
                lfn,
                readoutRecurrentError(doRnnReadout, lfn),
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
    innerLearner = create_learner(config.inner_learner, False, config.inner_uoro_std)
    innerLibrary = create_rnn_learner(innerLearner, lossFn)
    outerLearner = create_learner(config.outer_learner, True, config.outer_uoro_std)

    innerController = endowAveragedGradients(innerLibrary.modelGradient, config.tr_avg_per)
    innerController = logGradient(innerController)
    innerLibrary = innerLibrary._replace(modelGradient=innerController)

    inner_param, _ = innerInterpreter.getRecurrentParam.func(innerInterpreter, initEnv)
    outer_state, _ = outerInterpreter.getRecurrentState.func(outerInterpreter, initEnv)
    pad_val_grad_by = jnp.maximum(0, jnp.size(outer_state) - jnp.size(inner_param))

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
                pad_val_grad_by,
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
        tr, _ = innerLibrary.modelLossFn(oho_data.payload).func(innerInterpreter, env)

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
            outerInfluenceTensorNorm=jnp.linalg.norm(jnp.ravel(env.outerLogs.influenceTensor)),
            innerInfluenceTensorNorm=jnp.linalg.norm(jnp.ravel(env.innerLogs.influenceTensor)),
            largest_jacobian_eigenvalue=env.innerLogs.jac_eigenvalue,
            largest_hessian_eigenvalue=env.outerLogs.jac_eigenvalue,
        )
        return pure(log, PX[tuple[GodInterpreter, GodState]]())

    model = eqx.filter_jit(traverseM(updateStep)(dataset).func).lower(outerInterpreter, initEnv).compile()

    logs, trained_env = model(outerInterpreter, initEnv)
    return logs, trained_env


if __name__ == "__main__":
    main()
