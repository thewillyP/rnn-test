# %%
import argparse
import time
from typing import Any

import jax.experimental
import optax
from wandb.sdk.wandb_config import Config
from recurrent.monad import foldM
from recurrent.mylearning import *
from recurrent.myrecords import GodConfig, GodInterpreter
from recurrent.mytypes import *


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
import wandb

# jax.config.update("jax_enable_x64", True)


# def main():
#     # Initialize a new wandb run
#     with wandb.init(mode="offline"):
#         # If called by wandb.agent, as below,
#         # this config will be set by Sweep Controller
#         # config: Config = wandb.config
#         config = GodConfig(**wandb.config)
#         prng = PRNG(jax.random.key(config.seed))
#         test_prng = PRNG(jax.random.key(config.test_seed))

#         lossFn = getLossFn(config)
#         env, innerInterpreter, outerInterpreter = create_env(config, prng)


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


def test_create_env_with_recurrent(config: GodConfig, prng: PRNG) -> None:
    """
    Test function to verify create_env with an equinox-jitted function using
    recurrent state and recurrent parameters through the inner interpreter.

    Args:
        config: GodConfig object containing the environment configuration
        prng: PRNG object for random number generation
    """
    # Create the environment
    env, inner_interpreter, outer_interpreter = create_env(config, prng)

    # Define a jitted function that uses recurrent state and parameters

    def test_recurrent_step(
        state: GodState, interpreter: GodInterpreter, input_vec: jax.Array
    ) -> tuple[GodState, jax.Array]:
        """
        Performs one step using recurrent state and vectorized recurrent parameters.

        Args:
            state: Current GodState
            interpreter: Inner GodInterpreter
            input_vec: Input vector for this step

        Returns:
            Tuple of updated state and loss
        """
        # Get recurrent state and vectorized parameters using the interpreter
        rec_state_app = interpreter.getReccurentState
        rec_param_app = interpreter.getRecurrentParam  # Note: Assuming this exists or using getRnnParameter

        # Execute the monadic operations
        rec_state, _ = rec_state_app.func(interpreter, state)
        rec_params_flat, _ = rec_param_app.func(interpreter, state)

        # Reshape flat parameters back into usable form
        n_h, n_in = config.n_h, config.n_in
        total_param_size = n_h * (n_h + n_in + 1)  # w_rec size
        w_rec_flat = rec_params_flat[:total_param_size]
        W_rec = w_rec_flat.reshape(n_h, n_h + n_in + 1)

        W_rec_matrix = W_rec[:, :n_h]  # Recurrent weights
        W_in = W_rec[:, n_h : n_h + n_in]  # Input weights
        b_rec = W_rec[:, -1]  # Bias

        # Compute new recurrent state
        # Shape: (n_h,) = (n_h, n_h) @ (n_h,) + (n_h, n_in) @ (n_in,) + (n_h,)
        new_rec_state = jax.nn.tanh(W_rec_matrix @ rec_state + W_in @ input_vec + b_rec)

        # Update state using the interpreter
        update_state_app = interpreter.putReccurentState(new_rec_state)
        _, new_state = update_state_app.func(interpreter, state)

        # Compute a dummy loss (difference between new and old state)
        loss = jnp.mean(jnp.square(new_rec_state - rec_state))

        return new_state, loss

    # Test input vector
    input_vec = jax.random.normal(jax.random.PRNGKey(123), (config.n_in,))

    # Run the test
    initial_state = env
    initial_rec_state, _ = inner_interpreter.getReccurentState.func(inner_interpreter, initial_state)
    initial_rec_params, _ = inner_interpreter.getRecurrentParam.func(inner_interpreter, initial_state)

    print("Initial recurrent state shape:", initial_rec_state)
    print("Initial recurrent parameters shape:", initial_rec_params)
    print("Input vector shape:", input_vec.shape)

    # Execute the jitted function
    new_state, loss = eqx.filter_jit(test_recurrent_step)(initial_state, inner_interpreter, input_vec)

    new_rec_state, _ = inner_interpreter.getReccurentState.func(inner_interpreter, new_state)
    new_rec_params, _ = inner_interpreter.getRecurrentParam.func(inner_interpreter, new_state)

    print("New recurrent state shape:", new_rec_state)
    print("New recurrent parameters shape:", new_rec_params)
    print("Computed loss:", loss)

    # Verify the state has been updated
    assert not jnp.array_equal(initial_rec_state, new_rec_state), "Recurrent state should have changed after update"

    # Verify shapes are consistent
    assert initial_rec_state.shape == new_rec_state.shape, "Recurrent state shapes should remain consistent"
    assert initial_rec_params.shape == new_rec_params.shape, "Recurrent parameter shapes should remain consistent"

    # Verify parameters remain unchanged
    assert jnp.array_equal(initial_rec_params, new_rec_params), "Recurrent parameters should not change in this test"

    print("Test with recurrent state and recurrent parameters completed successfully!")


# Example usage
if __name__ == "__main__":
    # Create a sample config for testing
    sample_config = GodConfig(
        inner_learning_rate=0.01,
        outer_learning_rate=0.01,
        ts=(15, 17),  # Example tuple for time steps
        seed=42,
        test_seed=43,
        tr_examples_per_epoch=1000,
        vl_examples_per_epoch=200,
        tr_avg_per=10,
        vl_avg_per=5,
        numVal=200,
        numTr=1000,
        numTe=500,
        inner_learner="rflo",
        outer_learner="rflo",
        lossFn="cross_entropy",
        inner_optimizer="sgd",
        outer_optimizer="sgd",
        activation_fn="tanh",
        architecture="rnn",
        n_h=10,
        n_in=2,
        n_out=2,
        inner_time_constant=1.0,
        outer_time_constant=1.0,
        logFlag=True,
    )

    # Create PRNG
    test_prng = PRNG(jax.random.PRNGKey(42))

    # Run the test
    test_create_env_with_recurrent(sample_config, test_prng)

# # def create_env(config: GodConfig, prng: PRNG) -> GodState:
# #     def getter[T](f: Callable[[GodState], T]) -> Agent[GodInterpreter, T]:
# #         return get(PX[GodState]()).fmap(f)

# #     def putter[T](f: Callable[[GodState], T]) -> Callable[[T], Agent[GodInterpreter, Unit]]:
# #         return lambda x: modifies(lambda s: eqx.tree_at(f, s, x))

# #     prng1, prng2, prng3, prng4, prng5 = jax.random.split(prng, 5)

# #     logConfig = LogConfig(doLog=config.logFlag)

# #     match config.activation_fn:
# #         case "tanh":
# #             activationFn = jax.nn.tanh
# #         case "relu":
# #             activationFn = jax.nn.relu
# #         case _:
# #             raise ValueError("Invalid activation function")

# #     match config.architecture:
# #         case "rnn":
# #             W_in = jax.random.normal(prng2, (config.n_h, config.n_in)) * jnp.sqrt(1 / config.n_in)
# #             W_rec, _ = jnp.linalg.qr(jax.random.normal(prng3, (config.n_h, config.n_h)))  # QR decomposition
# #             W_out = jax.random.normal(prng4, (config.n_out, config.n_h)) * jnp.sqrt(1 / config.n_h)
# #             b_rec = jnp.zeros((config.n_h, 1))
# #             b_out = jnp.zeros((config.n_out, 1))

# #             w_rec = jnp.hstack([W_rec, W_in, b_rec])
# #             w_out = jnp.hstack([W_out, b_out])

# #             rnnParameter = RnnParameter(w_rec=w_rec, w_out=w_out)

# #             rnnConfig = RnnConfig(
# #                 n_h=config.n_h,
# #                 n_in=config.n_in,
# #                 n_out=config.n_out,
# #                 activationFn=activationFn,
# #             )

# #             activation = ACTIVATION(jax.random.normal(prng1, (config.n_h,)))
# #             rnnState = RnnState(rnnConfig=rnnConfig, activation=activation, rnnParameter=rnnParameter)

# #             rec_state_n = jnp.size(toVector(endowVector(activation)))
# #             rec_param_n = jnp.size(toVector(endowVector(rnnParameter)))

# #             innerInfluenceTensor = jnp.zeros((rec_state_n, rec_param_n))

# #             innerUoroInit = toVector(
# #                 endowVector(
# #                     RnnParameter(
# #                         w_rec=jax.random.normal(prng5, rnnParameter.w_rec.shape),
# #                         w_out=jnp.zeros_like(rnnParameter.w_out),
# #                     )
# #                 )
# #             )

# #             innerLogs = Logs(
# #                 gradient=jnp.zeros_like(toVector(endowVector(rnnParameter))),
# #                 influenceTensor=innerInfluenceTensor,
# #                 immediateInfluenceTensor=innerInfluenceTensor,
# #                 hessian=jnp.zeros((rec_state_n, rec_state_n)),
# #             )

# #             getActivationM = getter(lambda s: s.rnnState.activation)
# #             putActivationM = putter(lambda s: s.rnnState.activation)
# #             getRnnConfigM = getter(lambda s: s.rnnState.rnnConfig)
# #             getRnnParameterM = getter(lambda s: s.rnnState.rnnParameter)
# #             putRnnParameterM = putter(lambda s: s.rnnState.rnnParameter)
# #             getInfluenceTensorM = getter(lambda s: s.innerInfluenceTensor)
# #             putInfluenceTensorM = putter(lambda s: s.innerInfluenceTensor)

# #             getRecurrentState = getActivationM.fmap(compose2(endowVector, toVector))
# #             putRecurrentState = lambda x: getActivationM.fmap(lambda s: invmap(s, lambda _: x))
# #             getRecurrentParameter = getRnnParameterM.fmap(compose2(endowVector, toVector))
# #             putRecurrentParameter = lambda x: getRnnParameterM.fmap(lambda s: invmap(s, lambda _: x))

# #         case _:
# #             raise ValueError("Invalid architecture")

# #     match config.inner_optimizer:
# #         case "sgd":
# #             sgd = SgdParameter(learning_rate=config.inner_learning_rate)
# #             getOptimizerM = getter(lambda s: s.innerSgdParameter).fmap(lambda s: optax.sgd(s.learning_rate))
# #             getUpdaterM = pure(optax.apply_updates, PX[tuple[GodInterpreter, GodState]]())
# #             innerOptState = optax.sgd(sgd.learning_rate).init(jnp.empty((rec_param_n,)))
# #         case "sgd_positive":
# #             sgd = SgdParameter(learning_rate=config.inner_learning_rate)
# #             getOptimizerM = getter(lambda s: s.innerSgdParameter).fmap(lambda s: optax.sgd(s.learning_rate))

# #             def positive_updates(params: optax.Params, updates: optax.Updates):
# #                 return jax.tree.map(lambda p, u: jnp.maximum(p + u, 0), params, updates)

# #             getUpdaterM = pure(positive_updates, PX[tuple[GodInterpreter, GodState]]())
# #             innerOptState = optax.sgd(sgd.learning_rate).init(jnp.empty((rec_param_n,)))
# #         case _:
# #             raise ValueError("Invalid inner optimizer")

# #     match config.inner_learner:
# #         case "rtrl":
# #             getUoroM = getter(lambda s: s.innerUoro).flat_map(
# #                 lambda x: app_nothing(x, PX[tuple[GodInterpreter, GodState]]())
# #             )
# #             putUoroM = lambda _: app_nothing(Unit(), PX[tuple[GodInterpreter, GodState]]())
# #         case "uoro":
# #             getUoroM = getter(lambda s: s.innerUoro)
# #             putUoroM = putter(lambda s: s.innerUoro)
# #         case "rflo":
# #             getUoroM = getter(lambda s: s.innerUoro).flat_map(
# #                 lambda x: app_nothing(x, PX[tuple[GodInterpreter, GodState]]())
# #             )
# #             putUoroM = lambda _: app_nothing(Unit(), PX[tuple[GodInterpreter, GodState]]())
# #         case "identity":
# #             getUoroM = getter(lambda s: s.innerUoro).flat_map(
# #                 lambda x: app_nothing(x, PX[tuple[GodInterpreter, GodState]]())
# #             )
# #             putUoroM = lambda _: app_nothing(Unit(), PX[tuple[GodInterpreter, GodState]]())
# #         case _:
# #             raise ValueError("Invalid inner learner")

# #     innerInterpreter = GodInterpreter(
# #         getRecurrentState=getRecurrentState,
# #         putRecurrentState=putRecurrentState,
# #         getRecurrentParameter=getRecurrentParameter,
# #         putRecurrentParameter=putRecurrentParameter,
# #         getActivation=getActivationM,
# #         putActivation=putActivationM,
# #         getInfluenceTensor=getInfluenceTensorM,
# #         putInfluenceTensor=putInfluenceTensorM,
# #         getUoro=getUoroM,
# #         putUoro=putUoroM,
# #         getRnnConfig=getRnnConfigM,
# #         getTimeConstant=getter(lambda s: s.innerTimeConstant),
# #         getLogConfig=getter(lambda s: s.logConfig),
# #         putLogs=putter(lambda s: s.innerLogs),
# #         getRnnParameter=getRnnParameterM,
# #         putRnnParameter=putRnnParameterM,
# #         getOptState=getter(lambda s: s.innerOptState),
# #         putOptState=putter(lambda s: s.innerOptState),
# #         getOptimizer=getOptimizerM,
# #         getUpdater=getUpdaterM,
# #     )

# #     match config.outer_optimizer:
# #         case "sgd":
# #             outer_sgd = SgdParameter(learning_rate=config.outer_learning_rate)
# #             getOuterOptimizerM = getter(lambda s: s.outerSgdParameter).fmap(lambda s: optax.sgd(s.learning_rate))
# #             getOuterUpdaterM = pure(optax.apply_updates, PX[tuple[GodInterpreter, GodState]]())
# #             outerOptState = optax.sgd(outer_sgd.learning_rate).init(jnp.empty((rec_param_n,)))

# #             rec_state_n = jnp.size(toVector(endowVector(activation)))
# #             rec_param_n = jnp.size(toVector(endowVector(rnnParameter)))

# #         case "sgd_positive":
# #             outer_sgd = SgdParameter(learning_rate=config.outer_learning_rate)
# #             getOuterOptimizerM = getter(lambda s: s.outerSgdParameter).fmap(lambda s: optax.sgd(s.learning_rate))
# #             getOuterUpdaterM = pure(optax.apply_updates, PX[tuple[GodInterpreter, GodState]]())
# #             outerOptState = optax.sgd(outer_sgd.learning_rate).init(jnp.empty((rec_param_n,)))
# #         case _:
# #             raise ValueError("Invalid outer optimizer")


# def create_dataset(config: Config, prng: PRNG, test_prng: PRNG) -> Traversable[OhoData[InputOutput]]:
#     pass


# def generate_add_task_dataset(N, t_1, t_2, tau_task, rng_key):
#     """y(t) = 0.5 + 0.5 * x(t - t_1) - 0.25 * x(t - t_2)"""
#     N = N // tau_task

#     x = jax.random.bernoulli(rng_key, 0.5, (N,)).astype(jnp.float32)

#     y = 0.5 + 0.5 * jnp.roll(x, t_1) - 0.25 * jnp.roll(x, t_2)

#     X = jnp.asarray([x, 1 - x]).T
#     Y = jnp.asarray([y, 1 - y]).T

#     # Temporally stretch according to the desire timescale of change.
#     X = jnp.tile(X, tau_task).reshape((tau_task * N, 2))
#     Y = jnp.tile(Y, tau_task).reshape((tau_task * N, 2))

#     return X, Y


# def constructRnnEnv(rng_key: Array, initLearningRate: float):
#     """Constructs an RNN environment with predefined configurations."""

#     # Define network dimensions
#     n_h, n_in, n_out = 32, 2, 2
#     alpha = 1.0

#     # Define learning rates as arrays
#     # 0.02712261
#     learning_rate = jnp.asarray([initLearningRate])
#     meta_learning_rate = jnp.asarray([0.00])

#     rng_key, subkey1, subkey2, subkey3 = jax.random.split(rng_key, 4)

#     W_in = jax.random.normal(subkey1, (n_h, n_in)) * jnp.sqrt(1 / n_in)
#     W_rec, _ = jnp.linalg.qr(jax.random.normal(subkey2, (n_h, n_h)))  # QR decomposition
#     W_out = jax.random.normal(subkey3, (n_out, n_h)) * jnp.sqrt(1 / n_h)
#     b_rec = jnp.zeros((n_h, 1))
#     b_out = jnp.zeros((n_out, 1))

#     w_rec = jnp.hstack([W_rec, W_in, b_rec])
#     w_out = jnp.hstack([W_out, b_out])

#     # Initialize parameters
#     parameter = RnnParameter(w_rec=w_rec, w_out=w_out)
#     rnn_config = RnnConfig(n_h=n_h, n_in=n_in, n_out=n_out, alpha=alpha, activationFn=jnp.tanh)

#     sgd = SgdParameter(learning_rate=learning_rate)
#     sgd_mlr = SgdParameter(learning_rate=meta_learning_rate)

#     # Generate random activation state
#     rng_key, activation_rng = jax.random.split(rng_key)
#     activation = ACTIVATION(jax.random.normal(activation_rng, (n_h,)))

#     # Split keys for UORO
#     rng_key, prng_A, prng_B, prng_C = jax.random.split(rng_key, num=4)

#     # Construct environment state
#     num_activ = jnp.size(activation)
#     num_params = jnp.size(compose2(endowVector, toVector)(parameter))
#     influenceTensor_ = zeroedInfluenceTensor(n_h, parameter)
#     ohoInfluenceTensor_ = zeroedInfluenceTensor(jnp.size(compose2(endowVector, toVector)(parameter)), sgd)
#     init_env = RnnGodState[RnnParameter, SgdParameter, SgdParameter](
#         activation=activation,
#         influenceTensor=Jacobian[RnnParameter](influenceTensor_),
#         ohoInfluenceTensor=Jacobian[SgdParameter](ohoInfluenceTensor_),
#         parameter=parameter,
#         hyperparameter=sgd,
#         metaHyperparameter=sgd_mlr,
#         rnnConfig=rnn_config,
#         rnnConfig_bilevel=rnn_config,
#         uoro=UORO_Param(
#             A=jax.random.normal(prng_A, (n_h,)),
#             B=uoroBInit(prng_B, parameter),
#         ),
#         prng=prng_C,
#         logs=Logs(
#             gradient=jnp.zeros_like(toVector(endowVector(parameter))),
#             influenceTensor=influenceTensor_,
#             immediateInfluenceTensor=influenceTensor_,
#             hessian=jnp.zeros((num_activ, num_activ)),
#         ),
#         oho_logs=Logs(
#             gradient=jnp.zeros_like(toVector(endowVector(sgd))),
#             validationGradient=jnp.zeros_like(toVector(endowVector(parameter))),
#             influenceTensor=ohoInfluenceTensor_,
#             immediateInfluenceTensor=ohoInfluenceTensor_,
#             hessian=jnp.zeros((num_params, num_params)),
#         ),
#     )

#     return rng_key, init_env


# def train(
#     dataloader: Traversable[OhoInputOutput],
#     test_set: Traversable[InputOutput],
#     t_series_length: int,
#     trunc_length: int,
#     N_test: int,
#     N_val: int,
#     initEnv: RnnGodState[RnnParameter, SgdParameter, SgdParameter],
# ):
#     type Train_Dl = BaseRnnInterpreter[RnnParameter, SgdParameter, SgdParameter]
#     type OHO = OhoInterpreter[RnnParameter, SgdParameter, SgdParameter]
#     type ENV = RnnGodState[RnnParameter, SgdParameter, SgdParameter]

#     # interpreters
#     trainDialect = BaseRnnInterpreter[RnnParameter, SgdParameter, SgdParameter]()
#     ohoDialect = OhoInterpreter[RnnParameter, SgdParameter, SgdParameter](trainDialect)

#     # lambda key, shape: jax.random.uniform(key, shape, minval=-1.0, maxval=1.0)
#     onlineLearner: RnnLibrary[Train_Dl, InputOutput, ENV, PREDICTION, RnnParameter]
#     onlineLearner = RTRL[
#         InputOutput,
#         ENV,
#         ACTIVATION,
#         RnnParameter,
#         jax.Array,
#         PREDICTION,
#     ]().createLearner(
#         doRnnStep(),
#         doRnnReadout(),
#         lambda a, b: LOSS(optax.safe_softmax_cross_entropy(a, b)),
#         readoutRecurrentError(doRnnReadout(), lambda a, b: LOSS(optax.safe_softmax_cross_entropy(a, b))),
#     )

#     onlineLearner_folded = foldrRnnLearner(onlineLearner, initEnv.parameter)
#     rnnLearner = endowAveragedGradients(onlineLearner_folded, trunc_length, initEnv.parameter)
#     rnnLearner = logGradient(rnnLearner)
#     # rnnLearner = normalizeGradientRnnLibrary(rnnLearner)

#     oho_rtrl = RTRL[
#         OhoInputOutput,
#         ENV,
#         RnnParameter,
#         SgdParameter,
#         jax.Array,
#         Traversable[PREDICTION],
#     ]()
#     # oho_rtrl = IdentityLearner()

#     def validation_loss(label: Traversable[jax.Array], prediction: Traversable[PREDICTION]):
#         return LOSS(jnp.mean(optax.safe_softmax_cross_entropy(label.value, prediction.value)))

#     oho: RnnLibrary[OHO, OhoInputOutput, ENV, Traversable[PREDICTION], SgdParameter]
#     oho = endowBilevelOptimization(
#         rnnLearner,
#         doSgdStep,
#         trainDialect,
#         oho_rtrl,
#         validation_loss,
#         resetRnnActivation(initEnv.activation),
#     )
#     oho = logGradient(oho)
#     # oho = clipGradient(4.0, oho)

#     @do()
#     def updateStep():
#         env = yield from get(PX[ENV]())
#         interpreter = yield from askForInterpreter(PX[OHO]())
#         oho_data = yield from ask(PX[OhoInputOutput]())
#         validation_model = resetRnnActivation(initEnv.activation).then(rnnLearner.rnnWithLoss).func

#         learning_rate = yield from interpreter.getParameter().fmap(lambda x: x.learning_rate)
#         te, _ = validation_model(trainDialect, test_set, env)
#         vl, _ = validation_model(trainDialect, oho_data.validation, env)  # lag 1 timestep bc show prev param validation
#         tr, _ = rnnLearner.rnnWithLoss.func(trainDialect, oho_data.train, env)
#         weights = toVector(endowVector(env.parameter))

#         _ = yield from oho.rnnWithGradient.flat_map(doSgdStep_Positive)

#         log = AllLogs(
#             trainLoss=tr / t_series_length,
#             validationLoss=vl / N_val,
#             testLoss=te / N_test,
#             learningRate=learning_rate,
#             parameterNorm=jnp.linalg.norm(weights),
#             ohoGradient=env.oho_logs.gradient,
#             trainGradient=jnp.linalg.norm(env.logs.gradient),
#             validationGradient=jnp.linalg.norm(env.oho_logs.validationGradient),
#             immediateInfluenceTensor=jnp.linalg.norm(jnp.ravel(env.oho_logs.immediateInfluenceTensor)),
#             influenceTensor=jnp.linalg.norm(jnp.ravel(env.oho_logs.influenceTensor)),
#             hessian=jnp.linalg.eigvals(env.oho_logs.hessian),
#         )
#         return pure(log, PX3[OHO, OhoInputOutput, ENV]())

#     model = eqx.filter_jit(traverse(updateStep()).func).lower(ohoDialect, dataloader, initEnv).compile()

#     start = time.time()
#     logs, trained_env = model(ohoDialect, dataloader, initEnv)
#     tttt = f"Train Time: {time.time() - start}"
#     logs: AllLogs = logs.value

#     print(logs.trainLoss[-1])
#     print(logs.testLoss[-1])
#     print(logs.validationLoss[-1])
#     print(logs.learningRate[-1])
#     print(tttt)

#     return logs, trained_env


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
