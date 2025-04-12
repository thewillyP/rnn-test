import copy
import time
from typing import Any
import jax.numpy as jnp
import equinox as eqx
import jax
import wandb
from wandb import sdk as wandb_sdk
import uuid
from datetime import datetime
from torch.utils.data import DataLoader, Subset
import pickle
import os
import optax

from recurrent.mylearning import *
from recurrent.mylearning import RFLO, RTRL, UORO, IdentityLearner
from recurrent.myrecords import GodConfig, GodInterpreter, GodState
from recurrent.mytypes import *
from recurrent.parameters import (
    RnnConfig,
    RnnParameter,
    SgdParameter,
    UORO_Param,
)
from recurrent.pytree_dataset import PyTreeDataset, jax_collate_fn
from recurrent.util import *


def runApp(
    load_env: Callable[[GodConfig, PRNG], GodState],
    load_config: Callable[[wandb_sdk.wandb_run.Run], dict[str, Any]],
    wandb_kwargs: dict[str, Any],
):
    with wandb.init(**wandb_kwargs) as run:
        config = GodConfig(**load_config(run))
        prng = PRNG(jax.random.key(config.seed))
        test_prng = PRNG(jax.random.key(config.test_seed))
        env_prng, data_prng = jax.random.split(prng, 2)
        lossFn = getLossFn(config)
        checkpoint_fn = lambda env: save_checkpoint(env, f"trained_env_{run.id}", "env.pkl")

        env = load_env(config, env_prng)
        _, innerInterpreter, outerInterpreter = create_env(config, env_prng)
        oho_set, test_set = create_datasets(config, data_prng, test_prng)

        checkpoint_fn(copy.replace(env, prng=jax.random.key_data(env.prng)))
        start = time.time()
        train_loop_IO(oho_set, test_set, lossFn, env, innerInterpreter, outerInterpreter, config, checkpoint_fn)
        end = time.time()
        print(f"Training time: {end - start} seconds")


def generate_unique_id():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")  # Microsecond precision
    short_uuid = uuid.uuid4().hex[:8]  # Random component
    return f"{timestamp}_{short_uuid}"


def save_object_as_wandb_artifact(obj: Any, artifact_name: str, filename: str, artifact_type: str) -> None:
    """Pickle an arbitrary object and save it as a W&B artifact."""
    os.makedirs("artifacts", exist_ok=True)
    full_path = os.path.join("artifacts", filename)

    with open(full_path, "wb") as f:
        pickle.dump(obj, f)

    artifact = wandb.Artifact(artifact_name, type=artifact_type)
    artifact.add_file(full_path)
    wandb.log_artifact(artifact)
    print(f"Saved {artifact_name} as {filename}")


def save_checkpoint(obj: Any, name: str, filename: str) -> None:
    return save_object_as_wandb_artifact(obj, name, filename, "checkpoint")


def load_artifact(artifact_name: str, filename: str) -> Any:
    api = wandb.Api()
    model_artifact = api.artifact(artifact_name)
    model_dir = model_artifact.download()
    with open(os.path.join(model_dir, filename), "rb") as f:
        return pickle.load(f)


def log_jax_array(array: jax.Array, artifact_name: str):
    filename = f"{artifact_name}.npy"
    jnp.save(filename, array)
    artifact = wandb.Artifact(artifact_name, type="checkpoint")
    artifact.add_file(filename)
    wandb.log_artifact(artifact)
    os.remove(filename)


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
        innerTimeConstant=config.inner_time_constant,
        outerTimeConstant=config.outer_time_constant,
        start_epoch=0,
        start_example=0,
        globalLogConfig=GlobalLogConfig(stop_influence=False),
    )

    inner_log_config = LogConfig(
        log_special=config.inner_log_special,
        lanczos_iterations=config.inner_lanczos_iterations,
        log_expensive=config.inner_log_expensive if config.inner_log_expensive is not None else False,
    )
    env = putter(env, lambda s: s.innerLogConfig, inner_log_config)

    # 1) Initialize inner state and parameters

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
            rnn_config = RnnConfig(
                n_h=config.n_h, n_in=config.n_in, n_out=config.n_out, activationFn=config.activation_fn
            )
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
        case "sgd_clipped":
            sgd = SgdParameter(learning_rate=config.inner_learning_rate)
            opt_state = soft_clipped_sgd(
                config.inner_learning_rate, config.inner_clip, config.inner_clip_sharpness
            ).init(jnp.empty((rec_param_n,)))
            env = putter(env, lambda s: s.innerSgdParameter, sgd)
            env = putter(env, lambda s: s.innerOptState, opt_state)
            get_inner_optimizer = lambda s: soft_clipped_sgd(
                s.innerSgdParameter.learning_rate, config.inner_clip, config.inner_clip_sharpness
            )
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
        hessian=jnp.zeros((rec_state_n, rec_state_n)),
    )
    env = putter(env, lambda s: s.innerLogs, innerLogs)

    # =============================
    # 3) Initialize outer state and parameters

    outer_log_config = LogConfig(
        log_special=config.outer_log_special,
        lanczos_iterations=config.outer_lanczos_iterations,
        log_expensive=config.outer_log_expensive if config.outer_log_expensive is not None else False,
    )
    env = putter(env, lambda s: s.outerLogConfig, outer_log_config)

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
        case "sgd_clipped":
            sgd = SgdParameter(learning_rate=config.outer_learning_rate)
            opt_state = soft_clipped_sgd(
                config.outer_learning_rate, config.outer_clip, config.outer_clip_sharpness
            ).init(jnp.empty((outer_rec_param_n,)))
            env = putter(env, lambda s: s.outerSgdParameter, sgd)
            env = putter(env, lambda s: s.outerOptState, opt_state)
            get_outer_optimizer = lambda s: soft_clipped_sgd(
                s.outerSgdParameter.learning_rate, config.outer_clip, config.outer_clip_sharpness
            )
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
        hessian=jnp.zeros((outer_rec_state_n, outer_rec_state_n)),
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
        getLogConfig=monadic_getter(lambda s: s.innerLogConfig),
        getGlobalLogConfig=monadic_getter(lambda s: s.globalLogConfig),
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
        getLogConfig=monadic_getter(lambda s: s.outerLogConfig),
        getGlobalLogConfig=monadic_getter(lambda s: s.globalLogConfig),
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
        # when I add bptt clearly don't add foldrlibrary
        case _:
            library: Library[InputOutput, GodInterpreter, GodState, PREDICTION]
            library = learner.createLearner(
                lambda d: doRnnStep(d).then(ask(PX[GodInterpreter]())).flat_map(lambda i: i.getRecurrentState),
                doRnnReadout,  # these can be passed in dynamically later
                lfn,
                readoutRecurrentError(doRnnReadout, lfn),
            )
            return foldrLibrary(library)


def train_loop_IO(
    dataset: Traversable[OhoData[Traversable[InputOutput]]],
    test_dataset: Traversable[InputOutput],
    lossFn: Callable[[jax.Array, jax.Array], LOSS],
    initEnv: GodState,
    innerInterpreter: GodInterpreter,
    outerInterpreter: GodInterpreter,
    config: GodConfig,
    checkpoint_fn: Callable[[GodState], None],
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
        print("recompiled")
        env = yield from get(PX[GodState]())
        interpreter = yield from ask(PX[GodInterpreter]())
        hyperparameters = yield from interpreter.getRecurrentParam
        weights, _ = innerInterpreter.getRecurrentParam.func(innerInterpreter, env)

        validation_model = (
            lambda ds: resetRnnActivation(initEnv.rnnState.activation).then(innerLibrary.modelLossFn(ds)).func
        )

        te, _ = validation_model(test_dataset)(innerInterpreter, env)
        vl, _ = validation_model(oho_data.validation)(innerInterpreter, env)
        tr, _ = innerLibrary.modelLossFn(oho_data.payload).func(innerInterpreter, env)

        _ = yield from outerLibrary.modelGradient(oho_data).flat_map(doOptimizerStep)

        log = {
            "train_loss": tr / config.tr_examples_per_epoch,
            "validation_loss": vl / config.vl_examples_per_epoch,
            "test_loss": te / config.numTe,
            "hyperparameters": hyperparameters,
            "parameter_norm": jnp.linalg.norm(weights),
            "oho_gradient": env.outerLogs.gradient,
            "train_gradient": env.innerLogs.gradient,
            "validation_gradient": env.outerLogs.validationGradient,
            "immediate_influence_tensor_norm": jnp.linalg.norm(env.outerLogs.immediateInfluenceTensor),
            "outer_influence_tensor_norm": jnp.linalg.norm(env.outerLogs.influenceTensor),
            "outer_influence_tensor": env.outerLogs.influenceTensor,
            "inner_influence_tensor_norm": jnp.linalg.norm(env.innerLogs.influenceTensor),
            "largest_jacobian_eigenvalue": env.innerLogs.jac_eigenvalue,
            "largest_hessian_eigenvalue": env.outerLogs.jac_eigenvalue,
            "rnn_activation_norm": jnp.linalg.norm(env.rnnState.activation),
            "jacobian": env.innerLogs.hessian,
            "hessian": env.outerLogs.hessian,
            "immediate_influence_tensor": env.outerLogs.immediateInfluenceTensor,
        }
        return pure(log, PX[tuple[GodInterpreter, GodState]]())

    model = eqx.filter_jit(lambda d, e: traverseM(updateStep)(d).func(outerInterpreter, e))
    pytree_dataset = PyTreeDataset(dataset)
    env = initEnv

    num_batches_seen_so_far = 0
    all_logs = []
    for epoch in range(initEnv.start_epoch, config.num_retrain_loops):
        batched_dataset = Subset(pytree_dataset, indices=range(env.start_example, len(pytree_dataset)))
        dataloader = DataLoader(
            batched_dataset, batch_size=config.data_load_size, shuffle=False, collate_fn=jax_collate_fn
        )

        for i, batch in enumerate(dataloader):
            batch_size = len(jax.tree.leaves(batch)[0])
            logs, env = model(batch, env)
            env = eqx.tree_at(lambda t: t.start_example, env, env.start_example + batch_size)
            all_logs.append(logs)

            checkpoint_condition = num_batches_seen_so_far + i + 1
            if checkpoint_condition % config.checkpoint_interval == 0:
                checkpoint_fn(copy.replace(env, prng=jax.random.key_data(env.prng)))

        env = eqx.tree_at(lambda t: t.start_epoch, env, epoch + 1)
        env = eqx.tree_at(lambda t: t.start_example, env, 0)
        num_batches_seen_so_far += len(dataloader)

    flat_trees = [jax.tree_flatten(tree)[0] for tree in all_logs]
    total_logs = jax.tree.map(lambda *arrays: jnp.concatenate(arrays, axis=0), *flat_trees)

    # log wandb partial metrics
    for log_tree_ in tree_unstack_lazy(total_logs):
        log_data: AllLogs = jax.tree.map(
            lambda x: jnp.real(x) if x is not None and jnp.all(jnp.isfinite(x)) else None, log_tree_
        )
        wandb.log(
            {
                "train_loss": log_data.trainLoss,
                "validation_loss": log_data.validationLoss,
                "test_loss": log_data.testLoss,
                "hyperparameters": log_data.hyperparameters,
                "parameter_norm": log_data.parameterNorm,
                "oho_gradient": log_data.ohoGradient,
                "train_gradient": log_data.trainGradient,
                "validation_gradient": log_data.validationGradient,
                "immediate_influence_tensor_norm": log_data.immediateInfluenceTensorNorm,
                "inner_influence_tensor_norm": log_data.innerInfluenceTensorNorm,
                "outer_influence_tensor_norm": log_data.outerInfluenceTensorNorm,
                "largest_jacobian_eigenvalue": log_data.largest_jacobian_eigenvalue,
                "largest_influence_eigenvalue": log_data.largest_hessian_eigenvalue,
                "jacobian_eigenvalues": log_data.jacobian_eigenvalues,
                "rnn_activation_norm": log_data.rnn_activation_norm,
            }
        )

    save_object_as_wandb_artifact(total_logs, "logs", "logs.pkl", "logs")
    checkpoint_fn(copy.replace(env, prng=jax.random.key_data(env.prng)))
