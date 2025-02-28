from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Callable, Protocol
from donotation import do
import jax
from jax import Array
import jax.numpy as jnp
import equinox as eqx
from operator import add

import optax

from recurrent.monad import App, Maybe
from recurrent.myrecords import GodState, InputOutput, OhoData
from recurrent.mytypes import Gradient, Traversable
from recurrent.objectalgebra.typeclasses import *

from recurrent.mytypes import *
from recurrent.monad import *
from recurrent.parameters import *
from recurrent.util import get_leading_dim_size, pytree_split


type Agent[Interpreter, X] = App[Interpreter, GodState, X]
type Controller[Data, Interpreter, X] = Callable[[Data], Agent[Interpreter, X]]


type LossFn[Pred, Data] = Callable[[Pred, Data], LOSS]


class Library[Data, Interpreter, Pred](NamedTuple):
    model: Controller[Data, Interpreter, Pred]
    modelLossFn: Controller[Data, Interpreter, LOSS]
    modelGradient: Controller[Data, Interpreter, Gradient[REC_PARAM]]


class CreateLearner(Protocol):
    def createLearner[Data, Interpreter, Pred](
        self,
        activationStep: Controller[Data, Interpreter, REC_STATE],
        readoutStep: Controller[Data, Interpreter, Pred],
        lossFunction: LossFn[Pred, Data],
        lossGradientWrtActiv: Controller[Data, Interpreter, Gradient[REC_STATE]],
    ) -> Library[Data, Interpreter, Pred]: ...


@eqx.filter_jit
def jvp(f, primal, tangent):
    return eqx.filter_jvp(f, (primal,), (tangent,))[1]


@eqx.filter_jit
def jacobian_matrix_product(f, primal, matrix):
    wrapper = lambda p, t: jvp(f, p, t)
    return jax.vmap(wrapper, in_axes=(None, 1), out_axes=1)(primal, matrix)


def resetRnnActivation[Interpreter: PutActivation](resetActv: ACTIVATION) -> Agent[Interpreter, Unit]:
    return ask(PX[Interpreter]()).flat_map(lambda interpreter: interpreter.putActivation(resetActv))


class _SGD_Can(
    GetRecurrentParam,
    PutRecurrentParam,
    GetSgdParameter,
    Protocol,
): ...


@do()
def doSgdStep[Interpreter: _SGD_Can](gr: Gradient[REC_PARAM]) -> G[Agent[Interpreter, Unit]]:
    interpreter = yield from ask(PX[Interpreter]())
    isParam = yield from interpreter.getRecurrentParam
    hyperparam = yield from interpreter.getSgdParameter
    new_param = isParam - hyperparam.learning_rate * gr.value
    return interpreter.putRecurrentParam(REC_PARAM(new_param))


@do()
def doSgdStep_Squared[Interpreter: _SGD_Can](gr: Gradient[REC_PARAM]) -> G[Agent[Interpreter, Unit]]:
    interpreter = yield from ask(PX[Interpreter]())
    isParam = yield from interpreter.getRecurrentParam
    hyperparam = yield from interpreter.getSgdParameter
    new_param = isParam - (hyperparam.learning_rate**2) * gr.value
    return interpreter.putRecurrentParam(REC_PARAM(new_param))


@do()
def doSgdStep_Positive[Interpreter: _SGD_Can](gr: Gradient[REC_PARAM]) -> G[Agent[Interpreter, Unit]]:
    interpreter = yield from ask(PX[Interpreter]())
    isParam = yield from interpreter.getRecurrentParam
    hyperparam = yield from interpreter.getSgdParameter
    new_param = jnp.ravel(jnp.maximum(0, isParam - hyperparam.learning_rate * gr.value))
    return interpreter.putRecurrentParam(REC_PARAM(new_param))


@do()
def doSgdStep_Normalized[Interpreter: _SGD_Can](gr: Gradient[REC_PARAM]) -> G[Agent[Interpreter, Unit]]:
    interpreter = yield from ask(PX[Interpreter]())
    isParam = yield from interpreter.getRecurrentParam
    hyperparam = yield from interpreter.getSgdParameter
    grad_norm = jnp.linalg.norm(gr.value) + 1e-8  # Add epsilon to avoid division by zero
    normalized_grad = gr.value / grad_norm
    new_param = jnp.ravel(isParam - hyperparam.learning_rate * normalized_grad)
    return interpreter.putRecurrentParam(REC_PARAM(new_param))


@do()
def doExpGradStep[Interpreter: _SGD_Can](gr: Gradient[REC_PARAM]) -> G[Agent[Interpreter, Unit]]:
    interpreter = yield from ask(PX[Interpreter]())
    isParam = yield from interpreter.getRecurrentParam
    hyperparam = yield from interpreter.getSgdParameter

    # Apply exponentiated gradient update with sign correction
    new_param = jnp.ravel(isParam * jnp.exp(-hyperparam.learning_rate * jnp.sign(isParam) * gr.value))

    return interpreter.putRecurrentParam(REC_PARAM(new_param))


class _RnnActivation_Can(
    GetActivation,
    PutActivation,
    GetRnnParameter,
    GetRnnConfig,
    GetTimeConstant,
    Protocol,
): ...


@do()
def doRnnStep[Interpreter: _RnnActivation_Can](data: InputOutput) -> G[Agent[Interpreter, Unit]]:
    interpreter = yield from ask(PX[Interpreter]())
    a = yield from interpreter.getActivation
    param = yield from interpreter.getRnnParameter
    cfg = yield from interpreter.getRnnConfig
    alpha = yield from interpreter.getTimeConstant

    a_rec = param.w_rec @ jnp.concat((a, data.x, jnp.asarray([1.0])))
    a_new = ACTIVATION((1 - alpha) * a + alpha * cfg.activationFn(a_rec))
    return interpreter.putActivation(a_new)


class _RnnReadout_Can(
    GetActivation,
    GetRnnParameter,
    GetRnnConfig,
    Protocol,
): ...


@do()
def doRnnReadout[Interpreter: _RnnReadout_Can](_: InputOutput) -> G[Agent[Interpreter, PREDICTION]]:
    interpreter = yield from ask(PX[Interpreter]())
    a = yield from interpreter.getActivation
    param = yield from interpreter.getRnnParameter
    pred = PREDICTION(param.w_out @ jnp.concat((a, jnp.asarray([1.0]))))
    return pure(pred, PX[tuple[Interpreter, GodState]]())


# def doLoss[Interpreter, Pred, Data](lossFn: LossFn[Pred, Data], data: Data):  # -> Callable[..., App[Interpreter, Data, GodState, LOSS]]:
#     @do()
#     def _lossFn(pred: Pred):
#         interpreter = yield from ask(PX[Interpreter]())
#         loss = lossFn(pred, data)
#         return pure(loss, PX3[Interpreter, Data, GodState]())

#     return _lossFn


@do()
def doGradient[Interpreter, Wrt: jax.Array](
    model: Agent[Interpreter, LOSS],
    read_wrt: Agent[Interpreter, Wrt],
    write_wrt: Callable[[Wrt], Agent[Interpreter, Unit]],
) -> G[Agent[Interpreter, Gradient[Wrt]]]:
    interpreter = yield from ask(PX[Interpreter]())
    env = yield from get(PX[GodState]())
    param = yield from read_wrt

    def parametrized(p: Array) -> tuple[LOSS, tuple[Maybe[LOSS], GodState]]:
        maybe, s = write_wrt(p).then(model).func(interpreter, env)
        return maybe.payload, (maybe, s)

    grad: GRADIENT
    maybe: Maybe[LOSS]
    new_env: GodState
    grad, (maybe, new_env) = eqx.filter_jacrev(parametrized, has_aux=True)(param)

    _ = yield from put(new_env)
    return lift(maybe.fmap(lambda _: Gradient[Wrt](jnp.ravel(grad))), PX[tuple[Interpreter, GodState]]())


def endowAveragedGradients[Interpreter: GetRecurrentParam, Data](
    gradientFn: Controller[Traversable[Data], Interpreter, Gradient[REC_PARAM]],
    trunc: int,
) -> Controller[Traversable[Data], Interpreter, Gradient[REC_PARAM]]:
    @do()
    def avgGradient(dataset: Traversable[Data]) -> G[Agent[Interpreter, Gradient[REC_PARAM]]]:
        interpreter = yield from ask(PX[Interpreter]())
        param_shape: Gradient[REC_PARAM]
        param_shape = yield from interpreter.getRecurrentParam.fmap(lambda x: Gradient(jnp.zeros_like(x)))

        N = get_leading_dim_size(dataset)
        n_complete = (N // trunc) * trunc
        n_leftover = N - n_complete

        # Reshaping the dataset with 'trunc' makes scanning easier IF dataset is evenly shaped,
        # but JAX doesn’t allow jagged arrays. So, we adjust the data
        # to be evenly shaped for (scan . scan) and handle the rest separately.
        # ORDER MATTERS, need to scan first, then do the leftover, since they are contiguous activations
        ds_scannable_, ds_leftover = pytree_split(dataset, trunc)
        ds_scannable: Traversable[Traversable[Data]] = Traversable(ds_scannable_)

        gr_scanned = yield from accumulateM(
            gradientFn,
            add,
            param_shape,
        )(ds_scannable)

        env_after_scannable = yield from get(PX[GodState]())

        def ifLeftoverData(leftover: Traversable[Data]) -> tuple[Maybe[Gradient[REC_PARAM]], GodState]:
            avgLeftover: Callable[[Gradient[REC_PARAM]], Gradient[REC_PARAM]]
            avgLeftover = lambda gr: Gradient[REC_PARAM]((trunc / N) * gr_scanned.value + (n_leftover / N) * gr.value)
            return gradientFn(leftover).fmap(avgLeftover).func(interpreter, env_after_scannable)

        gradient_final: Maybe[Gradient[REC_PARAM]]
        env_final: GodState
        gradient_final, env_final = jax.lax.cond(
            n_leftover == 0,
            lambda _: (just(Gradient[REC_PARAM]((trunc / N) * gr_scanned.value)), env_after_scannable),
            ifLeftoverData,
            ds_leftover,
        )
        _ = yield from put(env_final)
        return lift(gradient_final, PX[tuple[Interpreter, GodState]]())

    return avgGradient


class _ReadoutRecurrentError_Can(
    GetRecurrentState,
    PutRecurrentState,
    Protocol,
): ...


def readoutRecurrentError[Data, Interpreter: _ReadoutRecurrentError_Can, Pred](
    predictionStep: Controller[Data, Interpreter, Pred], lossFunction: LossFn[Pred, Data]
) -> Controller[Data, Interpreter, Gradient[REC_STATE]]:
    @do()
    def _readoutRecurrentError(data: Data) -> G[Agent[Interpreter, Gradient[REC_STATE]]]:
        interpreter = yield from ask(PX[Interpreter]())
        willGetImmediateLoss = predictionStep(data).fmap(lambda p: lossFunction(p, data))
        return doGradient(
            willGetImmediateLoss,
            interpreter.getRecurrentState,
            interpreter.putRecurrentState,
        )

    return _readoutRecurrentError


# todo, recheck later
# @do()
# def doBatchGradients[Dl, D: Module | Array, E, Pr: Module](
#     step: Fold[Dl, D, E, Gradient[Pr]], e_dim: E
# ) -> G[Fold[Dl, D, E, Gradient[Pr]]]:
#     dl = yield from ProxyDl[Dl].askDl()
#     run: Callable[[D, E], tuple[Gradient[Pr], E]]
#     run = lambda d, e: step.func(dl, d, e)
#     gr = yield from toFold(jax.vmap(run, in_axes=(0, e_dim), out_axes=(0, e_dim)))
#     gr_summed: Pr
#     gr_summed = jax.tree.map(lambda x: jnp.sum(x, axis=0), gr.value)
#     return pure(Gradient(gr_summed))


class IdentityLearner:
    def createLearner[Data, Interpreter: GetRecurrentParam, Pred](
        self,
        activationStep: Controller[Data, Interpreter, REC_STATE],
        readoutStep: Controller[Data, Interpreter, Pred],
        lossFunction: LossFn[Pred, Data],
        _: Controller[Data, Interpreter, Gradient[REC_STATE]],
    ) -> Library[Data, Interpreter, Pred]:
        @do()
        def rnnWithGradient() -> G[Agent[Interpreter, Gradient[REC_PARAM]]]:
            interpreter = yield from ask(PX[Interpreter]())
            param_shape: Gradient[REC_PARAM]
            param_shape = yield from interpreter.getRecurrentParam.fmap(lambda x: Gradient(jnp.zeros_like(x)))
            return pure(param_shape, PX[tuple[Interpreter, GodState]]())

        return Library(
            model=lambda data: activationStep(data).then(readoutStep(data)),
            modelLossFn=lambda data: activationStep(data).then(readoutStep(data)).fmap(lambda p: lossFunction(p, data)),
            modelGradient=lambda data: activationStep(data).then(rnnWithGradient()),
        )


class OfflineLearning:
    class _OfflineLearner_Can(
        GetRecurrentParam,
        PutRecurrentParam,
        Protocol,
    ): ...

    def createLearner[Data, Interpreter: _OfflineLearner_Can, Pred](
        self,
        activationStep: Controller[Data, Interpreter, REC_STATE],
        readoutStep: Controller[Data, Interpreter, Pred],
        lossFunction: LossFn[Pred, Data],
        _: Controller[Data, Interpreter, Gradient[REC_STATE]],
    ) -> Library[Traversable[Data], Interpreter, Traversable[Pred]]:
        rnnStep = lambda data: activationStep(data).then(readoutStep(data))
        rnn2Loss = lambda data: rnnStep(data).fmap(lambda p: lossFunction(p, data))
        rnnWithLoss = accumulateM(rnn2Loss, add, LOSS(jnp.Array(0.0)))

        @do()
        def rnnWithGradient(data: Traversable[Data]) -> G[Agent[Interpreter, Gradient[REC_PARAM]]]:
            interpreter: Interpreter = yield from ask(PX[Interpreter]())
            return doGradient(rnnWithLoss(data), interpreter.getRecurrentParam, interpreter.putRecurrentParam)

        return Library(
            model=traverseM(rnnStep),
            modelLossFn=rnnWithLoss,
            modelGradient=rnnWithGradient,
        )


# technically this inheritance setup is wrong, these should be factories instead but I am a poor man
# the Interpreter typing is technically incorrect
class PastFacingLearn(ABC):
    @abstractmethod
    def creditAssignment[Interpreter](
        self,
        recurrentError: Agent[Interpreter, Gradient[REC_STATE]],
        activationStep: Agent[Interpreter, REC_STATE],
    ) -> Agent[Interpreter, Gradient[REC_PARAM]]: ...

    class _CreateModel_Can(PutRecurrentState, PutRecurrentParam, Protocol): ...

    @staticmethod
    def createRnnForward[Interpreter: _CreateModel_Can](activationStep: Agent[Interpreter, REC_STATE]):
        @do()
        def _creatingModel():
            interpreter = yield from ask(PX[Interpreter]())
            env = yield from get(PX[GodState]())

            def agentFn(actv: REC_STATE, param: REC_PARAM) -> tuple[REC_STATE, tuple[Maybe[REC_STATE], GodState]]:
                maybe, s = (
                    interpreter.putRecurrentState(actv)
                    .then(interpreter.putRecurrentParam(param))
                    .then(activationStep)
                    .func(interpreter, env)
                )
                return maybe.payload, (maybe, s)

            return pure(agentFn, PX[tuple[Interpreter, GodState]]())

        return _creatingModel()

    class _PastFacingLearner_Can(
        GetRecurrentState,
        PutRecurrentState,
        GetRecurrentParam,
        PutRecurrentParam,
        Protocol,
    ): ...

    def createLearner[Data, Interpreter: _PastFacingLearner_Can, Pred](
        self,
        activationStep: Controller[Data, Interpreter, REC_STATE],
        readoutStep: Controller[Data, Interpreter, Pred],
        lossFunction: LossFn[Pred, Data],
        lossGradientWrtActiv: Controller[Data, Interpreter, Gradient[REC_STATE]],
    ) -> Library[Data, Interpreter, Pred]:
        def immediateLoss(data: Data):
            return readoutStep(data).fmap(lambda p: lossFunction(p, data))

        @do()
        def rnnGradient(data: Data) -> G[Agent[Interpreter, Gradient[REC_PARAM]]]:
            interpreter = yield from ask(PX[Interpreter]())

            # order matters, need to update readout (grad_o) AFTER updating activation (grad_rec updates it)
            grad_rec = yield from self.creditAssignment(lossGradientWrtActiv(data), activationStep(data))
            grad_readout = yield from doGradient(
                immediateLoss(data),
                interpreter.getRecurrentParam,
                interpreter.putRecurrentParam,
            )

            return pure(grad_rec + grad_readout, PX[tuple[Interpreter, GodState]]())

        return Library(
            model=lambda data: activationStep(data).then(readoutStep(data)),
            modelLossFn=lambda data: activationStep(data).then(immediateLoss(data)),
            modelGradient=rnnGradient,
        )


class InfluenceTensorLearner(PastFacingLearn, ABC):
    class _InfluenceTensorLearner_Can(
        GetInfluenceTensor,
        PutInfluenceTensor,
        PastFacingLearn._PastFacingLearner_Can,
        PutLogs,
        Protocol,
    ): ...

    @abstractmethod
    def getInfluenceTensor[Interpreter](
        self, activationStep: Agent[Interpreter, REC_STATE]
    ) -> Agent[Interpreter, Jacobian[REC_PARAM]]: ...

    def creditAssignment[Interpreter: _InfluenceTensorLearner_Can](
        self,
        recurrentError: Agent[Interpreter, Gradient[REC_STATE]],
        activationStep: Agent[Interpreter, REC_STATE],
    ) -> Agent[Interpreter, Gradient[REC_PARAM]]:
        @do()
        def _creditAssignment() -> G[Agent[Interpreter, Gradient[REC_PARAM]]]:
            interpreter = yield from ask(PX[Interpreter]())
            influenceTensor = yield from self.getInfluenceTensor(activationStep)
            _ = yield from interpreter.putInfluenceTensor(JACOBIAN(influenceTensor.value))

            signal = yield from recurrentError  # ORDER matters
            recurrentGradient = Gradient[REC_PARAM](signal.value @ influenceTensor.value)

            log_influence = Logs(influenceTensor=influenceTensor.value)
            _ = yield from interpreter.putLogs(log_influence)
            return pure(recurrentGradient, PX[tuple[Interpreter, GodState]]())

        return _creditAssignment()


class RTRL(InfluenceTensorLearner):
    class _UpdateInfluence_Can(
        GetInfluenceTensor,
        GetRecurrentState,
        PutRecurrentState,
        GetRecurrentParam,
        PutRecurrentParam,
        PutLogs,
        GetLogConfig,
        Protocol,
    ): ...

    @do()
    def getInfluenceTensor[Interpreter: _UpdateInfluence_Can](self, activationStep: Agent[Interpreter, REC_STATE]):
        # Get interpreter and RNN forward function
        interpreter = yield from ask(PX[Interpreter]())
        rnnForward = yield from self.createRnnForward(activationStep)

        # Extract initial states and parameters
        influenceTensor = yield from interpreter.getInfluenceTensor
        actv0 = yield from interpreter.getRecurrentState
        param0 = yield from interpreter.getRecurrentParam

        # Define activation function for Jacobian computation
        wrtActvFn = lambda a: rnnForward(a, param0)[0]

        # Compute immediate Jacobian influence tensor product
        # Note: Using JVP for efficiency in forward-over-reverse computation
        # See: https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
        immediateJacobian__InfluenceTensor_product: Array = jacobian_matrix_product(
            wrtActvFn,
            actv0,
            influenceTensor,
        )

        # Compute immediate influence with auxiliary data
        immediateInfluence: Array
        maybe: Maybe[REC_STATE]
        env: GodState
        immediateInfluence, (maybe, env) = eqx.filter_jacfwd(lambda p: rnnForward(actv0, p), has_aux=True)(param0)

        # Construct new influence tensor
        newInfluenceTensor: Jacobian[REC_PARAM] = Jacobian(
            immediateJacobian__InfluenceTensor_product + immediateInfluence
        )

        # Conditional Jacobian computation based on logging config
        log_condition: bool = yield from interpreter.getLogConfig.fmap(lambda x: x.doLog)
        shape_info = jax.eval_shape(eqx.filter_jacrev(wrtActvFn), actv0)
        jacobian = jax.lax.cond(
            log_condition,
            lambda _: eqx.filter_jacrev(wrtActvFn)(actv0),
            lambda _: jax.tree_map(lambda x: jnp.zeros(x.shape, dtype=x.dtype), shape_info),
            None,
        )

        # Store results and return
        _ = yield from put(env)
        _ = yield from interpreter.putLogs(Logs(immediateInfluenceTensor=immediateInfluence))
        _ = yield from interpreter.putLogs(Logs(hessian=jacobian))
        return lift(maybe.fmap(lambda _: newInfluenceTensor), PX[tuple[Interpreter, GodState]]())


class RFLO(InfluenceTensorLearner):
    class _UpdateRFLO_Can(
        GetTimeConstant,
        GetInfluenceTensor,
        GetRecurrentState,
        PutRecurrentState,
        GetRecurrentParam,
        PutRecurrentParam,
        PutLogs,
        Protocol,
    ): ...

    @do()
    def getInfluenceTensor[Interpreter: _UpdateRFLO_Can](self, activationStep: Agent[Interpreter, REC_STATE]):
        interpreter = yield from ask(PX[Interpreter]())
        rnnForward = yield from self.createRnnForward(activationStep)

        alpha = yield from interpreter.getTimeConstant
        influenceTensor = yield from interpreter.getInfluenceTensor
        actv0 = yield from interpreter.getRecurrentState
        param0 = yield from interpreter.getRecurrentParam

        immediateInfluence: Array
        maybe: Maybe[REC_STATE]
        env: GodState
        immediateInfluence, (maybe, env) = eqx.filter_jacrev(lambda p: rnnForward(actv0, p), has_aux=True)(param0)

        newInfluenceTensor: Jacobian[REC_PARAM] = Jacobian((1 - alpha) * influenceTensor + alpha * immediateInfluence)

        _ = yield from put(env)
        _ = yield from interpreter.putLogs(Logs(immediateInfluenceTensor=immediateInfluence))
        return lift(maybe.fmap(lambda _: newInfluenceTensor), PX[tuple[Interpreter, GodState]]())


class UORO(PastFacingLearn):
    class _UORO_Can(
        PastFacingLearn._PastFacingLearner_Can,
        GetUoro,
        PutUoro,
        GetPRNG,
        Protocol,
    ): ...

    def __init__(self, distribution: Callable[[PRNG, tuple[int]], Array]):
        self.distribution = distribution

    def getProjectedGradients[Interpreter: _UORO_Can](
        self, randomVector: Array, activationStep: Agent[Interpreter, REC_STATE]
    ) -> Agent[Interpreter, tuple[Array, Array]]:
        @do()
        def projectGradients():
            interpreter = yield from ask(PX[Interpreter]())
            rnnForward = yield from self.createRnnForward(activationStep)

            uoro = yield from interpreter.getUoro
            actv0 = yield from interpreter.getRecurrentState
            param0 = yield from interpreter.getRecurrentParam

            # 1 calculate the actv_jacobian vector product with A
            immediateJacobian__A_projection: Array
            immediateJacobian__A_projection = jvp(
                lambda a: rnnForward(a, param0)[0],
                actv0,
                uoro.A,
            )

            # 2 doing the vjp saves BIG on memory. Only use O(n^2) as we want
            def fn(p: Array) -> tuple[REC_STATE, tuple[Maybe[REC_STATE], GodState]]:
                return rnnForward(actv0, p)

            _, vjp_func, (maybe, env) = eqx.filter_vjp(fn, param0, has_aux=True)
            immediateInfluence__Random_projection: Array
            (immediateInfluence__Random_projection,) = vjp_func(randomVector)

            assert isinstance(immediateJacobian__A_projection, Array)
            assert isinstance(immediateInfluence__Random_projection, Array)

            _ = yield from put(env)
            maybe_new = maybe.fmap(lambda _: (immediateJacobian__A_projection, immediateInfluence__Random_projection))
            return lift(maybe_new, PX[tuple[Interpreter, GodState]]())

        return projectGradients()

    def propagateRecurrentError[Interpreter: _UORO_Can](
        self, A_: Array, B_: Array, recurrentError: Agent[Interpreter, Gradient[REC_STATE]]
    ) -> Agent[Interpreter, Gradient[REC_PARAM]]:
        def propagate(signal: Gradient[REC_STATE]) -> Gradient[REC_PARAM]:
            q = signal.value @ A_
            return Gradient(q * B_)

        return recurrentError.fmap(propagate)

    def creditAssignment[Interpreter: _UORO_Can](
        self,
        recurrentError: Agent[Interpreter, Gradient[REC_STATE]],
        activationStep: Agent[Interpreter, REC_STATE],
    ) -> Agent[Interpreter, Gradient[REC_PARAM]]:
        @do()
        def creditAssignment_():
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

            return self.propagateRecurrentError(A_new, B_new, recurrentError)

        return creditAssignment_()


def foldrLibrary[Data, Interpreter: GetRecurrentParam, Pred](
    library: Library[Data, Interpreter, Pred],
) -> Library[Interpreter, Traversable[Data], Traversable[Pred]]:
    @do()
    def modelGradient_(data: Data):
        interpreter = yield from ask(PX[Interpreter]())
        param_shape: Gradient[REC_PARAM]
        param_shape = yield from interpreter.getRecurrentParam.fmap(lambda x: Gradient(jnp.zeros_like(x)))

        return accumulateM(library.modelGradient, add, param_shape)(data)

    return Library(
        model=traverseM(library.model),
        modelLossFn=accumulateM(library.modelLossFn, add, LOSS(jnp.array(0.0))),
        modelGradient=modelGradient_,
    )


def endowBilevelOptimization[
    OHO_Interpreter: PutLogs,
    TrainInterpreter: GetRecurrentParam,
    Data,
    Pred,
](
    library: Library[Data, TrainInterpreter, Pred],
    paramFn: Callable[[Gradient[REC_PARAM]], Agent[TrainInterpreter, Unit]],
    trainInterpreter: TrainInterpreter,
    bilevelOptimizer: CreateLearner,
    computeLoss: LossFn[Pred, OhoData[Data]],
    resetEnvForValidation: Agent[TrainInterpreter, Unit],
) -> Library[OhoData[Data], OHO_Interpreter, Pred]:
    @do()
    def newActivationStep(oho_data: OhoData[Data]) -> Agent[OHO_Interpreter, REC_PARAM]:
        @do()
        def updateParameter() -> G[Agent[TrainInterpreter, REC_PARAM]]:
            interpreter = yield from ask(PX[TrainInterpreter]())
            return library.modelGradient(oho_data.payload).flat_map(paramFn).then(interpreter.getRecurrentParam)

        return updateParameter().switch_dl(trainInterpreter)

    @do()
    def newPredictionStep(oho_data: OhoData[Data]) -> G[Agent[OHO_Interpreter, Pred]]:
        env = yield from get(PX[GodState]())
        agentValPr = library.model(oho_data.validation)
        predictions, _ = resetEnvForValidation.then(agentValPr).func(trainInterpreter, env)
        return lift(predictions, PX[tuple[OHO_Interpreter, GodState]]())

    @do()
    def newRecurrentErrorStep(oho_data: OhoData[Data]) -> G[Agent[OHO_Interpreter, Gradient[REC_PARAM]]]:
        env = yield from get(PX[GodState]())
        agentValGr = library.modelGradient(oho_data.validation)
        recurrentGradient_, _ = resetEnvForValidation.then(agentValGr).func(trainInterpreter, env)
        recurrentGradient = yield from lift(recurrentGradient_, PX[tuple[TrainInterpreter, GodState]]())

        validation_log = Logs(validationGradient=recurrentGradient.value)
        interpreter = yield from ask(PX[OHO_Interpreter]())
        _ = yield from interpreter.putLogs(validation_log)
        return pure(recurrentGradient, PX[tuple[OHO_Interpreter, GodState]]())

    return bilevelOptimizer.createLearner(newActivationStep, newPredictionStep, computeLoss, newRecurrentErrorStep)


def normalizeGradient[Data, Interpreter](
    agentGr: Controller[Data, Interpreter, Gradient[REC_PARAM]],
) -> Controller[Data, Interpreter, Gradient[REC_PARAM]]:
    def normalizeGradient(gr: Gradient[REC_PARAM]) -> Gradient[REC_PARAM]:
        grad_norm = jnp.linalg.norm(gr.value) + 1e-8  # Avoid division by zero
        return Gradient(gr.value / grad_norm)

    return lambda data: agentGr(data).fmap(normalizeGradient)


def clipGradient[Data, Interpreter](
    clip: float,
    agentGr: Controller[Data, Interpreter, Gradient[REC_PARAM]],
) -> Controller[Data, Interpreter, Gradient[REC_PARAM]]:
    def clippedGradient(gr: Gradient[REC_PARAM]) -> Gradient[REC_PARAM]:
        clipped_gr, _ = optax.clip_by_global_norm(clip).update(gr.value, optax.EmptyState())
        return Gradient(clipped_gr)

    return lambda data: agentGr(data).fmap(clippedGradient)


def logGradient[Data, Interpreter: PutLogs](
    agentGr: Controller[Data, Interpreter, Gradient[REC_PARAM]],
):
    @do()
    def _logGradient(data: Data):
        interpreter = yield from ask(PX[Interpreter]())
        gr = yield from agentGr(data)
        _ = yield from interpreter.putLogs(Logs(gradient=gr.value))
        return pure(gr, PX[tuple[Interpreter, GodState]]())

    return _logGradient
