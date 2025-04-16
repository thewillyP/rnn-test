from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Callable, Protocol
from donotation import do
import jax
from jax import Array
import jax.numpy as jnp
import equinox as eqx
from operator import add

import matfree
import matfree.decomp
import matfree.eig
import optax

from recurrent.monad import *
from recurrent.myrecords import GodState, InputOutput, OhoData
from recurrent.mytypes import Gradient, Traversable
from recurrent.objectalgebra.typeclasses import *
from recurrent.mytypes import *
from recurrent.monad import *
from recurrent.parameters import *
from recurrent.util import get_leading_dim_size, pytree_split

type Agent[Interpreter, Env, X] = App[Interpreter, Env, X]
type Controller[Data, Interpreter, Env, X] = Callable[[Data], Agent[Interpreter, Env, X]]

type LossFn[Pred, Data] = Callable[[Pred, Data], LOSS]


class Library[Data, Interpreter, Env, Pred](NamedTuple):
    model: Controller[Data, Interpreter, Env, Pred]
    modelLossFn: Controller[Data, Interpreter, Env, LOSS]
    modelGradient: Controller[Data, Interpreter, Env, Gradient[REC_PARAM]]


class CreateLearner(Protocol):
    def createLearner[Data, Interpreter, Env, Pred](
        self,
        activationStep: Controller[Data, Interpreter, Env, REC_STATE],
        readoutStep: Controller[Data, Interpreter, Env, Pred],
        lossFunction: LossFn[Pred, Data],
        lossGradientWrtActiv: Controller[Data, Interpreter, Env, Gradient[REC_STATE]],
    ) -> Library[Data | Traversable[Data], Interpreter, Env, Pred | Traversable[Pred]]: ...


@eqx.filter_jit
def jvp(f, primal, tangent):
    return eqx.filter_jvp(f, (primal,), (tangent,))[1]


@eqx.filter_jit
def jacobian_matrix_product(f, primal, matrix):
    wrapper = lambda p, t: jvp(f, p, t)
    return jax.vmap(wrapper, in_axes=(None, 1), out_axes=1)(primal, matrix)


def resetRnnActivation[Interpreter: PutActivation[Env], Env](resetActv: ACTIVATION) -> Agent[Interpreter, Env, Unit]:
    return ask(PX[Interpreter]()).flat_map(lambda interpreter: interpreter.putActivation(resetActv))


def map_activation_fn(actv_fn: str):
    match actv_fn:
        case "tanh":
            return jax.nn.tanh
        case "relu":
            return jax.nn.relu
        case _:
            raise ValueError("Invalid activation function")


class _Opt_Can[Env](
    GetRecurrentParam[Env],
    PutRecurrentParam[Env],
    GetOptimizer[Env],
    GetOptState[Env],
    PutOptState[Env],
    GetUpdater[Env],
    Protocol,
): ...


@do()
def doOptimizerStep[Interpreter: _Opt_Can[Env], Env](gr: Gradient[REC_PARAM]) -> G[Agent[Interpreter, Env, Unit]]:
    interpreter = yield from ask(PX[Interpreter]())
    param = yield from interpreter.getRecurrentParam
    optimizer = yield from interpreter.getOptimizer
    opt_state = yield from interpreter.getOptState
    updater = yield from interpreter.getUpdater
    updates, new_opt_state = optimizer.update(gr.value, opt_state, param)
    new_param = updater(param, updates)
    _ = yield from interpreter.putOptState(new_opt_state)
    return interpreter.putRecurrentParam(REC_PARAM(new_param))


def normalized_sgd(learning_rate):
    return optax.chain(
        optax.normalize_by_update_norm(scale_factor=1.0),
        optax.sgd(learning_rate),
    )


def soft_clip_norm(threshold: float, sharpness: float):
    def update_fn(updates, state, _):
        grads_flat, unravel_fn = jax.flatten_util.ravel_pytree(updates)
        grad_norm = jnp.linalg.norm(grads_flat)
        clipped_norm = grad_norm - jax.nn.softplus(sharpness * (grad_norm - threshold)) / sharpness
        scale = clipped_norm / (grad_norm + 1e-6)
        updates_scaled = jax.tree_util.tree_map(lambda g: g * scale, updates)
        return updates_scaled, state

    return optax.GradientTransformation(lambda _: (), update_fn)


def soft_clipped_sgd(learning_rate, threshold, sharpness):
    return optax.chain(
        soft_clip_norm(threshold, sharpness),
        optax.sgd(learning_rate),
    )


class _RnnActivation_Can[Env](
    GetActivation[Env],
    PutActivation[Env],
    GetRnnParameter[Env],
    GetRnnConfig[Env],
    GetTimeConstant[Env],
    Protocol,
): ...


@do()
def doRnnStep[Interpreter: _RnnActivation_Can[Env], Env](data: InputOutput) -> G[Agent[Interpreter, Env, Unit]]:
    interpreter = yield from ask(PX[Interpreter]())
    a = yield from interpreter.getActivation
    param = yield from interpreter.getRnnParameter
    cfg = yield from interpreter.getRnnConfig
    alpha = yield from interpreter.getTimeConstant

    a_rec = param.w_rec @ jnp.concat((a, data.x, jnp.asarray([1.0])))
    a_new = ACTIVATION((1 - alpha) * a + alpha * map_activation_fn(cfg.activationFn)(a_rec))
    return interpreter.putActivation(a_new)


class _RnnReadout_Can[Env](
    GetActivation[Env],
    GetRnnParameter[Env],
    GetRnnConfig[Env],
    Protocol,
): ...


@do()
def doRnnReadout[Interpreter: _RnnReadout_Can[Env], Env](_: InputOutput) -> G[Agent[Interpreter, Env, PREDICTION]]:
    interpreter = yield from ask(PX[Interpreter]())
    a = yield from interpreter.getActivation
    param = yield from interpreter.getRnnParameter
    pred = PREDICTION(param.w_out @ jnp.concat((a, jnp.asarray([1.0]))))
    return pure(pred, PX[tuple[Interpreter, Env]]())


@do()
def doGradient[Interpreter, Env, Wrt: jax.Array](
    model: Agent[Interpreter, Env, LOSS],
    read_wrt: Agent[Interpreter, Env, Wrt],
    write_wrt: Callable[[Wrt], Agent[Interpreter, Env, Unit]],
) -> G[Agent[Interpreter, Env, Gradient[Wrt]]]:
    interpreter = yield from ask(PX[Interpreter]())
    env = yield from get(PX[Env]())
    param = yield from read_wrt

    def parametrized(p: Array) -> tuple[LOSS, Env]:
        return write_wrt(p).then(model).func(interpreter, env)

    grad, new_env = eqx.filter_jacrev(parametrized, has_aux=True)(param)
    _ = yield from put(new_env)
    return pure(Gradient[Wrt](jnp.ravel(grad)), PX[tuple[Interpreter, Env]]())


def endowAveragedGradients[Interpreter: GetRecurrentParam[Env], Env, Data](
    gradientFn: Controller[Traversable[Data], Interpreter, Env, Gradient[REC_PARAM]],
    trunc: int,
) -> Controller[Traversable[Data], Interpreter, Env, Gradient[REC_PARAM]]:
    @do()
    def avgGradient(dataset: Traversable[Data]) -> G[Agent[Interpreter, Env, Gradient[REC_PARAM]]]:
        interpreter = yield from ask(PX[Interpreter]())
        param_shape: Gradient[REC_PARAM]
        param_shape = yield from interpreter.getRecurrentParam.fmap(lambda x: Gradient(jnp.zeros_like(x)))

        N = get_leading_dim_size(dataset)
        n_complete = (N // trunc) * trunc
        n_leftover = N - n_complete

        ds_scannable_, ds_leftover = pytree_split(dataset, trunc)
        ds_scannable: Traversable[Traversable[Data]] = Traversable(ds_scannable_)

        gr_scanned = yield from accumulateM(gradientFn, add, param_shape)(ds_scannable)
        env_after_scannable = yield from get(PX[Env]())

        def if_leftover_data(leftover: Traversable[Data]) -> tuple[Gradient[REC_PARAM], Env]:
            gr, new_env = gradientFn(leftover).func(interpreter, env_after_scannable)
            avg = Gradient[REC_PARAM]((trunc / N) * gr_scanned.value + (n_leftover / N) * gr.value)
            return avg, new_env

        def no_leftover(_: Traversable[Data]) -> tuple[Gradient[REC_PARAM], Env]:
            return Gradient[REC_PARAM]((trunc / N) * gr_scanned.value), env_after_scannable

        gradient_final, env_final = jax.lax.cond(
            n_leftover > 0,
            if_leftover_data,
            no_leftover,
            ds_leftover,
        )
        _ = yield from put(env_final)
        return pure(gradient_final, PX[tuple[Interpreter, Env]]())

    return avgGradient


class _ReadoutRecurrentError_Can[Env](
    GetRecurrentState[Env],
    PutRecurrentState[Env],
    Protocol,
): ...


def readoutRecurrentError[Data, Interpreter: _ReadoutRecurrentError_Can[Env], Env, Pred](
    predictionStep: Controller[Data, Interpreter, Env, Pred], lossFunction: LossFn[Pred, Data]
) -> Controller[Data, Interpreter, Env, Gradient[REC_STATE]]:
    @do()
    def _readoutRecurrentError(data: Data) -> G[Agent[Interpreter, Env, Gradient[REC_STATE]]]:
        interpreter = yield from ask(PX[Interpreter]())
        willGetImmediateLoss = predictionStep(data).fmap(lambda p: lossFunction(p, data))
        return doGradient(willGetImmediateLoss, interpreter.getRecurrentState, interpreter.putRecurrentState)

    return _readoutRecurrentError


class IdentityLearner:
    def createLearner[Data, Interpreter: GetRecurrentParam[Env], Env, Pred](
        self,
        activationStep: Controller[Data, Interpreter, Env, REC_STATE],
        readoutStep: Controller[Data, Interpreter, Env, Pred],
        lossFunction: LossFn[Pred, Data],
        _: Controller[Data, Interpreter, Env, Gradient[REC_STATE]],
    ) -> Library[Data, Interpreter, Env, Pred]:
        @do()
        def rnnWithGradient() -> G[Agent[Interpreter, Env, Gradient[REC_PARAM]]]:
            interpreter = yield from ask(PX[Interpreter]())
            param_shape: Gradient[REC_PARAM]
            param_shape = yield from interpreter.getRecurrentParam.fmap(lambda x: Gradient(jnp.zeros_like(x)))
            return pure(param_shape, PX[tuple[Interpreter, Env]]())

        return Library(
            model=lambda data: activationStep(data).then(readoutStep(data)),
            modelLossFn=lambda data: activationStep(data).then(readoutStep(data)).fmap(lambda p: lossFunction(p, data)),
            modelGradient=lambda data: activationStep(data).then(rnnWithGradient()),
        )


class OfflineLearning:
    class _OfflineLearner_Can[Env](
        GetRecurrentParam[Env],
        PutRecurrentParam[Env],
        Protocol,
    ): ...

    def createLearner[Data, Interpreter: _OfflineLearner_Can[Env], Env, Pred](
        self,
        activationStep: Controller[Data, Interpreter, Env, REC_STATE],
        readoutStep: Controller[Data, Interpreter, Env, Pred],
        lossFunction: LossFn[Pred, Data],
        _: Controller[Data, Interpreter, Env, Gradient[REC_STATE]],
    ) -> Library[Traversable[Data], Interpreter, Env, Traversable[Pred]]:
        rnnStep = lambda data: activationStep(data).then(readoutStep(data))
        rnn2Loss = lambda data: rnnStep(data).fmap(lambda p: lossFunction(p, data))
        rnnWithLoss = accumulateM(rnn2Loss, add, LOSS(jnp.array(0.0)))

        @do()
        def rnnWithGradient(data: Traversable[Data]) -> G[Agent[Interpreter, Env, Gradient[REC_PARAM]]]:
            interpreter: Interpreter = yield from ask(PX[Interpreter]())
            return doGradient(rnnWithLoss(data), interpreter.getRecurrentParam, interpreter.putRecurrentParam)

        return Library(
            model=traverseM(rnnStep),
            modelLossFn=rnnWithLoss,
            modelGradient=rnnWithGradient,
        )


class PastFacingLearn(ABC):
    @abstractmethod
    def creditAssignment[Interpreter, Env](
        self,
        recurrentError: Agent[Interpreter, Env, Gradient[REC_STATE]],
        activationStep: Agent[Interpreter, Env, REC_STATE],
    ) -> Agent[Interpreter, Env, Gradient[REC_PARAM]]: ...

    class _CreateModel_Can[Env](PutRecurrentState[Env], PutRecurrentParam[Env], Protocol): ...

    @staticmethod
    def createRnnForward[Interpreter: _CreateModel_Can[Env], Env](activationStep: Agent[Interpreter, Env, REC_STATE]):
        @do()
        def _creatingModel():
            interpreter = yield from ask(PX[Interpreter]())
            env = yield from get(PX[Env]())

            def agentFn(actv: REC_STATE, param: REC_PARAM) -> tuple[REC_STATE, Env]:
                return (
                    interpreter.putRecurrentState(actv)
                    .then(interpreter.putRecurrentParam(param))
                    .then(activationStep)
                    .func(interpreter, env)
                )

            return pure(agentFn, PX[tuple[Interpreter, Env]]())

        return _creatingModel()

    class _PastFacingLearner_Can[Env](
        GetRecurrentState[Env],
        PutRecurrentState[Env],
        GetRecurrentParam[Env],
        PutRecurrentParam[Env],
        Protocol,
    ): ...

    def createLearner[Data, Interpreter: _PastFacingLearner_Can[Env], Env, Pred](
        self,
        activationStep: Controller[Data, Interpreter, Env, REC_STATE],
        readoutStep: Controller[Data, Interpreter, Env, Pred],
        lossFunction: LossFn[Pred, Data],
        lossGradientWrtActiv: Controller[Data, Interpreter, Env, Gradient[REC_STATE]],
    ) -> Library[Data, Interpreter, Env, Pred]:
        def immediateLoss(data: Data):
            return readoutStep(data).fmap(lambda p: lossFunction(p, data))

        @do()
        def rnnGradient(data: Data) -> G[Agent[Interpreter, Env, Gradient[REC_PARAM]]]:
            interpreter = yield from ask(PX[Interpreter]())

            grad_rec = yield from self.creditAssignment(lossGradientWrtActiv(data), activationStep(data))
            grad_readout = yield from doGradient(
                immediateLoss(data),
                interpreter.getRecurrentParam,
                interpreter.putRecurrentParam,
            )

            return pure(grad_rec + grad_readout, PX[tuple[Interpreter, Env]]())

        return Library(
            model=lambda data: activationStep(data).then(readoutStep(data)),
            modelLossFn=lambda data: activationStep(data).then(immediateLoss(data)),
            modelGradient=rnnGradient,
        )


class InfluenceTensorLearner(PastFacingLearn, ABC):
    class _InfluenceTensorLearner_Can[Env](
        GetInfluenceTensor[Env],
        PutInfluenceTensor[Env],
        PastFacingLearn._PastFacingLearner_Can[Env],
        PutLogs[Env],
        GetGlobalLogConfig[Env],
        Protocol,
    ): ...

    @abstractmethod
    def getInfluenceTensor[Interpreter, Env](
        self, activationStep: Agent[Interpreter, Env, REC_STATE]
    ) -> Agent[Interpreter, Env, Jacobian[REC_PARAM]]: ...

    def creditAssignment[Interpreter: _InfluenceTensorLearner_Can[Env], Env](
        self,
        recurrentError: Agent[Interpreter, Env, Gradient[REC_STATE]],
        activationStep: Agent[Interpreter, Env, REC_STATE],
    ) -> Agent[Interpreter, Env, Gradient[REC_PARAM]]:
        @do()
        def _creditAssignment() -> G[Agent[Interpreter, Env, Gradient[REC_PARAM]]]:
            interpreter = yield from ask(PX[Interpreter]())
            infT = yield from self.getInfluenceTensor(activationStep)
            stop_influence = yield from interpreter.getGlobalLogConfig.fmap(lambda x: x.stop_influence)
            if not stop_influence:
                _ = yield from interpreter.putInfluenceTensor(JACOBIAN(infT.value))

            influenceTensor = yield from interpreter.getInfluenceTensor
            signal = yield from recurrentError
            recurrentGradient = Gradient[REC_PARAM](signal.value @ influenceTensor)

            _ = yield from interpreter.putLogs(Logs(influenceTensor=influenceTensor))
            return pure(recurrentGradient, PX[tuple[Interpreter, Env]]())

        return _creditAssignment()


class RTRL(InfluenceTensorLearner):
    class _UpdateInfluence_Can[Env](
        GetInfluenceTensor[Env],
        GetRecurrentState[Env],
        PutRecurrentState[Env],
        GetRecurrentParam[Env],
        PutRecurrentParam[Env],
        PutLogs[Env],
        GetLogConfig[Env],
        GetPRNG[Env],
        Protocol,
    ): ...

    def __init__(self, use_fwd: bool):
        self.immediateJacFn = eqx.filter_jacfwd if use_fwd else eqx.filter_jacrev

    @do()
    def getInfluenceTensor[Interpreter: _UpdateInfluence_Can[Env], Env](
        self, activationStep: Agent[Interpreter, Env, REC_STATE]
    ) -> G[Agent[Interpreter, Env, Jacobian[REC_PARAM]]]:
        interpreter = yield from ask(PX[Interpreter]())
        rnnForward = yield from self.createRnnForward(activationStep)
        influenceTensor = yield from interpreter.getInfluenceTensor
        actv0 = yield from interpreter.getRecurrentState
        param0 = yield from interpreter.getRecurrentParam

        wrtActvFn = lambda a: rnnForward(a, param0)[0]
        immediateJacobian__InfluenceTensor_product: Array = jacobian_matrix_product(wrtActvFn, actv0, influenceTensor)
        immediateInfluence: Array
        env: Env
        immediateInfluence, env = self.immediateJacFn(lambda p: rnnForward(actv0, p), has_aux=True)(param0)
        newInfluenceTensor = Jacobian[REC_PARAM](immediateJacobian__InfluenceTensor_product + immediateInfluence)

        log_condition = yield from interpreter.getLogConfig.fmap(lambda x: x.log_special)
        lanczos_iterations = yield from interpreter.getLogConfig.fmap(lambda x: x.lanczos_iterations)
        log_expensive = yield from interpreter.getLogConfig.fmap(lambda x: x.log_expensive)
        subkey = yield from interpreter.updatePRNG()

        if log_condition:
            v0: Array = jnp.array(jax.random.normal(subkey, actv0.shape))
            tridag = matfree.decomp.tridiag_sym(lanczos_iterations, custom_vjp=False)
            get_eig = matfree.eig.eigh_partial(tridag)
            fn = lambda v: jvp(lambda a: wrtActvFn(a), actv0, v)
            eigvals, _ = get_eig(fn, v0)
            _ = yield from interpreter.putLogs(Logs(jac_eigenvalue=jnp.max(eigvals)))

        if log_expensive:
            _ = yield from interpreter.putLogs(Logs(hessian=eqx.filter_jacrev(wrtActvFn)(actv0)))

        _ = yield from put(env)
        _ = yield from interpreter.putLogs(Logs(immediateInfluenceTensor=immediateInfluence))
        return pure(newInfluenceTensor, PX[tuple[Interpreter, Env]]())


class RFLO(InfluenceTensorLearner):
    class _UpdateRFLO_Can[Env](
        GetTimeConstant[Env],
        GetInfluenceTensor[Env],
        GetRecurrentState[Env],
        PutRecurrentState[Env],
        GetRecurrentParam[Env],
        PutRecurrentParam[Env],
        PutLogs[Env],
        Protocol,
    ): ...

    @do()
    def getInfluenceTensor[Interpreter: _UpdateRFLO_Can[Env], Env](
        self, activationStep: Agent[Interpreter, Env, REC_STATE]
    ) -> G[Agent[Interpreter, Env, Jacobian[REC_PARAM]]]:
        interpreter = yield from ask(PX[Interpreter]())
        rnnForward = yield from self.createRnnForward(activationStep)
        alpha = yield from interpreter.getTimeConstant
        influenceTensor = yield from interpreter.getInfluenceTensor
        actv0 = yield from interpreter.getRecurrentState
        param0 = yield from interpreter.getRecurrentParam

        immediateInfluence: Array
        env: Env
        immediateInfluence, env = eqx.filter_jacrev(lambda p: rnnForward(actv0, p), has_aux=True)(param0)
        newInfluenceTensor = Jacobian[REC_PARAM]((1 - alpha) * influenceTensor + alpha * immediateInfluence)

        _ = yield from put(env)
        _ = yield from interpreter.putLogs(Logs(immediateInfluenceTensor=immediateInfluence))
        return pure(newInfluenceTensor, PX[tuple[Interpreter, Env]]())


class UORO(PastFacingLearn):
    class _UORO_Can[Env](
        PastFacingLearn._PastFacingLearner_Can[Env],
        GetUoro[Env],
        PutUoro[Env],
        GetPRNG[Env],
        PutLogs[Env],
        GetLogConfig[Env],
        Protocol,
    ): ...

    def __init__(self, distribution: Callable[[PRNG, tuple[int]], Array]):
        self.distribution = distribution

    def getProjectedGradients[Interpreter: _UORO_Can[Env], Env](
        self, randomVector: Array, activationStep: Agent[Interpreter, Env, REC_STATE]
    ) -> Agent[Interpreter, Env, tuple[Array, Array]]:
        @do()
        def projectGradients():
            interpreter = yield from ask(PX[Interpreter]())
            rnnForward = yield from self.createRnnForward(activationStep)

            uoro = yield from interpreter.getUoro
            actv0 = yield from interpreter.getRecurrentState
            param0 = yield from interpreter.getRecurrentParam

            immediateJacobian__A_projection: Array
            immediateJacobian__A_projection = jvp(
                lambda a: rnnForward(a, param0)[0],
                actv0,
                uoro.A,
            )

            def fn(p: Array) -> tuple[REC_STATE, Env]:
                return rnnForward(actv0, p)

            _, vjp_func, env = eqx.filter_vjp(fn, param0, has_aux=True)
            immediateInfluence__Random_projection: Array
            (immediateInfluence__Random_projection,) = vjp_func(randomVector)

            # assert isinstance(immediateJacobian__A_projection, Array)
            # assert isinstance(immediateInfluence__Random_projection, Array)

            _ = yield from put(env)
            pair = (immediateJacobian__A_projection, immediateInfluence__Random_projection)
            return pure(pair, PX[tuple[Interpreter, Env]]())

        return projectGradients()

    def propagateRecurrentError[Interpreter: _UORO_Can[Env], Env](
        self, A_: Array, B_: Array, recurrentError: Agent[Interpreter, Env, Gradient[REC_STATE]]
    ) -> Agent[Interpreter, Env, Gradient[REC_PARAM]]:
        def propagate(signal: Gradient[REC_STATE]) -> Gradient[REC_PARAM]:
            q = signal.value @ A_
            return Gradient(q * B_)

        return recurrentError.fmap(propagate)

    def creditAssignment[Interpreter: _UORO_Can[Env], Env](
        self,
        recurrentError: Agent[Interpreter, Env, Gradient[REC_STATE]],
        activationStep: Agent[Interpreter, Env, REC_STATE],
    ) -> Agent[Interpreter, Env, Gradient[REC_PARAM]]:
        @do()
        def creditAssignment_() -> G[Agent[Interpreter, Env, Gradient[REC_PARAM]]]:
            interpreter = yield from ask(PX[Interpreter]())
            uoro = yield from interpreter.getUoro
            key = yield from interpreter.updatePRNG()
            state_shape = yield from interpreter.getRecurrentState.fmap(jnp.shape)
            randomVector = self.distribution(key, state_shape)
            B_prev = uoro.B

            (
                immediateJacobian__A_projection,
                immediateInfluence__Random_projection,
            ) = yield from self.getProjectedGradients(randomVector, activationStep)

            rho0 = jnp.sqrt(jnp.linalg.norm(B_prev) / jnp.linalg.norm(immediateJacobian__A_projection))
            rho1 = jnp.sqrt(jnp.linalg.norm(immediateInfluence__Random_projection) / jnp.linalg.norm(randomVector))

            A_new = rho0 * immediateJacobian__A_projection + rho1 * randomVector
            B_new = B_prev / rho0 + immediateInfluence__Random_projection / rho1
            _ = yield from interpreter.putUoro(replace(uoro, A=A_new, B=B_new))

            log_condition = yield from interpreter.getLogConfig.fmap(lambda x: x.log_special)
            if log_condition:
                _ = yield from interpreter.putLogs(Logs(influenceTensor=jnp.outer(A_new, B_new)))

            return self.propagateRecurrentError(A_new, B_new, recurrentError)

        return creditAssignment_()


def foldrLibrary[Data, Interpreter: GetRecurrentParam[Env], Env, Pred](
    library: Library[Data, Interpreter, Env, Pred],
) -> Library[Traversable[Data], Interpreter, Env, Traversable[Pred]]:
    @do()
    def modelGradient_(data: Traversable[Data]) -> G[Agent[Interpreter, Env, Gradient[REC_PARAM]]]:
        interpreter = yield from ask(PX[Interpreter]())
        param_shape: Gradient[REC_PARAM]
        param_shape = yield from interpreter.getRecurrentParam.fmap(lambda x: Gradient(jnp.zeros_like(x)))

        return accumulateM(library.modelGradient, add, param_shape)(data)

    return Library(
        model=traverseM(library.model),
        modelLossFn=accumulateM(library.modelLossFn, add, LOSS(jnp.array(0.0))),
        modelGradient=modelGradient_,
    )


class _EndowBilevelOptimization_Can[Env](
    GetRecurrentState[Env],
    PutLogs[Env],
    GetPRNG[Env],
    Protocol,
): ...


def endowBilevelOptimization[
    OHO_Interpreter: _EndowBilevelOptimization_Can[Env],
    TrainInterpreter,
    Env,
    Data,
    Pred,
](
    library: Library[Data, TrainInterpreter, Env, Pred],
    paramFn: Callable[[Gradient[REC_PARAM]], Agent[TrainInterpreter, Env, Unit]],
    trainInterpreter: TrainInterpreter,
    bilevelOptimizer: CreateLearner,
    computeLoss: LossFn[Pred, OhoData[Data]],
    resetEnvForValidation: Callable[[Env, PRNG], Env],
    grad_add_pad_by: int,
) -> (
    Library[OhoData[Data], OHO_Interpreter, Env, Pred]
    | Library[Traversable[OhoData[Data]], OHO_Interpreter, Env, Traversable[Pred]]
):
    @do()
    def newActivationStep(oho_data: OhoData[Data]) -> G[Agent[OHO_Interpreter, Env, REC_PARAM]]:
        interpreter = yield from ask(PX[OHO_Interpreter]())
        return (
            library.modelGradient(oho_data.payload)
            .flat_map(paramFn)
            .switch_dl(trainInterpreter)
            .then(interpreter.getRecurrentState)
        )

    @do()
    def newPredictionStep(oho_data: OhoData[Data]) -> G[Agent[OHO_Interpreter, Env, Pred]]:
        env = yield from get(PX[Env]())
        interpreter = yield from ask(PX[OHO_Interpreter]())
        prng = yield from interpreter.updatePRNG()

        predictions, _ = library.model(oho_data.validation).func(trainInterpreter, resetEnvForValidation(env, prng))
        return pure(predictions, PX[tuple[OHO_Interpreter, Env]]())

    @do()
    def newRecurrentErrorStep(oho_data: OhoData[Data]) -> G[Agent[OHO_Interpreter, Env, Gradient[REC_PARAM]]]:
        env = yield from get(PX[Env]())
        interpreter = yield from ask(PX[OHO_Interpreter]())
        prng = yield from interpreter.updatePRNG()

        agentValGr = library.modelGradient(oho_data.validation)
        recurrentGradient, _ = agentValGr.func(trainInterpreter, resetEnvForValidation(env, prng))

        _ = yield from interpreter.putLogs(Logs(validationGradient=recurrentGradient.value))

        # Pad the gradient to match the influence tensor size
        # assumes first subsection corresponds to unilevel parameters
        # TODO: To not have to make this assumption, I need to be able to choose what I deriv wrt, instead of always just model parameters.
        recGrad_pad = jnp.concatenate([recurrentGradient.value, jnp.zeros((grad_add_pad_by,))])
        return pure(Gradient(recGrad_pad), PX[tuple[OHO_Interpreter, Env]]())

    return bilevelOptimizer.createLearner(newActivationStep, newPredictionStep, computeLoss, newRecurrentErrorStep)


def normalizeGradient[Data, Interpreter, Env](
    agentGr: Controller[Data, Interpreter, Env, Gradient[REC_PARAM]],
) -> Controller[Data, Interpreter, Env, Gradient[REC_PARAM]]:
    def normalizeGradient(gr: Gradient[REC_PARAM]) -> Gradient[REC_PARAM]:
        grad_norm = jnp.linalg.norm(gr.value) + 1e-8
        return Gradient(gr.value / grad_norm)

    return lambda data: agentGr(data).fmap(normalizeGradient)


def clipGradient[Data, Interpreter, Env](
    clip: float,
    agentGr: Controller[Data, Interpreter, Env, Gradient[REC_PARAM]],
) -> Controller[Data, Interpreter, Env, Gradient[REC_PARAM]]:
    def clippedGradient(gr: Gradient[REC_PARAM]) -> Gradient[REC_PARAM]:
        clipped_gr, _ = optax.clip_by_global_norm(clip).update(gr.value, optax.EmptyState())
        return Gradient(clipped_gr)

    return lambda data: agentGr(data).fmap(clippedGradient)


def logGradient[Data, Interpreter: PutLogs[Env], Env](
    agentGr: Controller[Data, Interpreter, Env, Gradient[REC_PARAM]],
) -> Controller[Data, Interpreter, Env, Gradient[REC_PARAM]]:
    @do()
    def _logGradient(data: Data) -> G[Agent[Interpreter, Env, Gradient[REC_PARAM]]]:
        interpreter = yield from ask(PX[Interpreter]())
        gr = yield from agentGr(data)
        _ = yield from interpreter.putLogs(Logs(gradient=gr.value))
        return pure(gr, PX[tuple[Interpreter, Env]]())

    return _logGradient
