from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Callable, Protocol
from donotation import do
import jax
from jax import Array
import jax.numpy as jnp
import equinox as eqx
from equinox import Module
from operator import add

import optax

from recurrent.monad import App, Maybe
from recurrent.myfunc import compose2
from recurrent.myrecords import GodState, MlApp
from recurrent.mytypes import Gradient, Traversable
from recurrent.objectalgebra.typeclasses import *

from recurrent.mytypes import *
from recurrent.monad import *
from recurrent.parameters import *
from recurrent.util import get_leading_dim_size, pytree_split, pytreeNumel


type Agent[Interpreter, X] = App[Interpreter, GodState, X]
type Controller[Data, Interpreter, X] = Callable[[Data], Agent[Interpreter, X]]


type LossFn[Pred, Data] = Callable[[Pred, Data], LOSS]


class Library[Data, Interpreter, Pred](NamedTuple):
    model: Controller[Data, Interpreter, Pred]
    modelLossFn: Controller[Data, Interpreter, LOSS]
    modelGradient: Controller[Data, Interpreter, Gradient[REC_PARAM]]


class CreateLearner[Data, Interpreter, Pred](Protocol):
    def createLearner(
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
    Protocol,
): ...


@do()
def doRnnStep[Interpreter: _RnnActivation_Can](data: InputOutput) -> G[Agent[Interpreter, Unit]]:
    interpreter = yield from ask(PX[Interpreter]())
    a = yield from interpreter.getActivation
    param = yield from interpreter.getRnnParameter
    cfg = yield from interpreter.getRnnConfig

    a_rec = param.w_rec @ jnp.concat((a, data.x, jnp.asarray([1.0])))
    a_new = ACTIVATION((1 - cfg.alpha) * a + cfg.alpha * cfg.activationFn(a_rec))
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


def readoutRecurrentError[Interpreter: _ReadoutRecurrentError_Can, Data, Pred](
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


class IdentityLearner[Data, Interpreter: GetRecurrentParam, Pred]:
    def createLearner(
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

    def createLearner[Interpreter: _OfflineLearner_Can, Data, Pred](
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

        xdf = traverseM(rnnStep)
        return Library(
            model=traverseM(rnnStep),
            modelLossFn=rnnWithLoss,
            modelGradient=rnnWithGradient,
        )


class PastFacingLearn[Data, Pred]:
    @abstractmethod
    def creditAssignment[Interpreter](
        self,
        recurrentError: MlApp[Interpreter, Data, Gradient[REC_STATE]],
        activationStep: MlApp[Interpreter, Data, REC_STATE],
    ) -> MlApp[Interpreter, Data, Gradient[REC_PARAM]]: ...

    class _CreateModel_Can(PutRecurrentState, PutRecurrentParam, Protocol): ...

    @staticmethod
    def createRnnForward[Interpreter: _CreateModel_Can](
        activationStep: MlApp[Interpreter, Data, REC_STATE],
    ):
        @do()
        def _creatingModel():
            interpreter = yield from askForInterpreter(PX[Interpreter]())
            data = yield from ask(PX[Data]())
            env = yield from get(PX[GodState]())

            def parametrized(actv_vec: REC_STATE, param_vec: REC_PARAM):
                maybe, s = (
                    interpreter.putRecurrentState(actv_vec)
                    .then(interpreter.putRecurrentParam(param_vec))
                    .then(activationStep)
                    .func(interpreter, data, env)
                )
                return maybe.payload, (maybe, s)

            return pure(parametrized, PX3[Interpreter, Data, GodState]())

        return _creatingModel()

    class _PastFacingLearner_Can(
        GetRecurrentState,
        PutRecurrentState,
        GetRecurrentParam,
        PutRecurrentParam,
        GetLabel,
        Protocol,
    ): ...

    def createLearner[Interpreter: _PastFacingLearner_Can](
        self,
        activationStep: MlApp[Interpreter, Data, REC_STATE],
        readoutStep: MlApp[Interpreter, Data, Pred],
        lossFunction: LossFn[Pred],
        lossGradientWrtActiv: MlApp[Interpreter, Data, Gradient[REC_STATE]],
    ):
        willGetImmediateLoss = readoutStep.flat_map(doLoss(lossFunction))

        @do()
        def rnnGradient() -> G[MlApp[Interpreter, Data, Gradient[REC_PARAM]]]:
            interpreter = yield from askForInterpreter(PX[Interpreter]())

            # order matters, need to update readout (grad_o) AFTER updating activation (grad_rec updates it)
            grad_rec = yield from self.creditAssignment(lossGradientWrtActiv, activationStep)
            grad_readout = yield from doGradient(
                willGetImmediateLoss,
                interpreter.getRecurrentParam,
                interpreter.putRecurrentParam,
            )

            return pure(grad_rec + grad_readout, PX3[Interpreter, Data, GodState]())

        return Library(
            model=activationStep.then(readoutStep),
            modelLossFn=activationStep.then(willGetImmediateLoss),
            modelGradient=activationStep.then(rnnGradient()),
        )


class InfluenceTensorLearner[Data, Pred](PastFacingLearn[Data, Pred]):
    class _InfluenceTensorLearner_Can(
        GetInfluenceTensor,
        PutInfluenceTensor,
        PastFacingLearn[Data, Pred]._PastFacingLearner_Can,
        PutLogs,
        Protocol,
    ): ...

    @abstractmethod
    def getInfluenceTensor[Interpreter](
        self, activationStep: MlApp[Interpreter, Data, REC_STATE]
    ) -> MlApp[Interpreter, Data, Jacobian[REC_PARAM]]: ...

    def creditAssignment[Interpreter: _InfluenceTensorLearner_Can](
        self,
        recurrentError: MlApp[Interpreter, Data, Gradient[REC_STATE]],
        activationStep: MlApp[Interpreter, Data, REC_STATE],
    ) -> App[Interpreter, Data, GodState, Gradient[REC_PARAM]]:
        @do()
        def _creditAssignment():
            interpreter = yield from askForInterpreter(PX[Interpreter]())
            influenceTensor = yield from self.getInfluenceTensor(activationStep)
            signal = yield from recurrentError
            recurrentGradient = Gradient[REC_PARAM](signal.value @ influenceTensor.value)
            _ = yield from interpreter.putInfluenceTensor(JACOBIAN(influenceTensor.value))

            log_influence = Logs(influenceTensor=influenceTensor.value)
            _ = yield from interpreter.putLogs(log_influence)
            return pure(recurrentGradient, PX3[Interpreter, Data, GodState]())

        return _creditAssignment()


class RTRL[Data, Pred](InfluenceTensorLearner[Data, Pred]):
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
    def getInfluenceTensor[Interpreter: _UpdateInfluence_Can](
        self, activationStep: MlApp[Interpreter, Data, REC_STATE]
    ):
        # Get interpreter and RNN forward function
        interpreter = yield from askForInterpreter(PX[Interpreter]())
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
        return lift(maybe.fmap(lambda _: newInfluenceTensor), PX3[Interpreter, Data, GodState]())


class RFLO[Data, Pred](InfluenceTensorLearner[Data, Pred]):
    class _UpdateRFLO_Can(
        GetRnnConfig,
        GetInfluenceTensor,
        GetRecurrentState,
        PutRecurrentState,
        GetRecurrentParam,
        PutRecurrentParam,
        Protocol,
    ): ...

    @do()
    def getInfluenceTensor[Interpreter: _UpdateRFLO_Can](self, activationStep: MlApp[Interpreter, Data, REC_STATE]):
        interpreter = yield from askForInterpreter(PX[Interpreter]())
        rnnForward = yield from self.createRnnForward(activationStep)

        alpha = yield from interpreter.getRnnConfig.fmap(lambda x: x.alpha)
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
        return lift(maybe.fmap(lambda _: newInfluenceTensor), PX3[Interpreter, Data, GodState]())


class UORO[Data, Pred](PastFacingLearn[Data, Pred]):
    class _UORO_Can(
        PastFacingLearn[Data, Pred]._PastFacingLearner_Can,
        GetUoro,
        PutUoro,
        GetRnnConfig,
        GetPRNG,
        Protocol,
    ): ...

    def __init__(self, distribution: Callable[[PRNG, tuple[int]], Array]):
        self.distribution = distribution

    def getProjectedGradients[Interpreter: _UORO_Can](
        self, randomVector: Array, activationStep: MlApp[Interpreter, Data, REC_STATE]
    ) -> App[Interpreter, Data, GodState, tuple[Array, Array]]:
        @do()
        def projectGradients():
            interpreter = yield from askForInterpreter(PX[Interpreter]())
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
            fn: Callable[[Array], tuple[Array, tuple[Maybe[REC_STATE], GodState]]] = lambda p: rnnForward(actv0, p)
            _, vjp_func, (maybe, env) = eqx.filter_vjp(fn, param0, has_aux=True)
            immediateInfluence__Random_projection: Array
            (immediateInfluence__Random_projection,) = vjp_func(randomVector)

            assert isinstance(immediateJacobian__A_projection, Array)
            assert isinstance(immediateInfluence__Random_projection, Array)

            _ = yield from put(env)
            maybe_new = maybe.fmap(lambda _: (immediateJacobian__A_projection, immediateInfluence__Random_projection))
            return lift(maybe_new, PX3[Interpreter, Data, GodState]())

        return projectGradients()

    def propagateRecurrentError[Interpreter: _UORO_Can](
        self, A_: Array, B_: Array, recurrentError: MlApp[Interpreter, Data, Gradient[REC_STATE]]
    ) -> App[Interpreter, Data, GodState, Gradient[REC_PARAM]]:
        def propagate(signal: Gradient[REC_STATE]) -> Gradient[REC_PARAM]:
            q = signal.value @ A_
            return Gradient(q * B_)

        return recurrentError.fmap(propagate)

    def creditAssignment[Interpreter: _UORO_Can](
        self,
        recurrentError: MlApp[Interpreter, Data, Gradient[REC_STATE]],
        activationStep: MlApp[Interpreter, Data, REC_STATE],
    ) -> App[Interpreter, Data, GodState, Gradient[REC_PARAM]]:
        @do()
        def creditAssignment_():
            interpreter = yield from askForInterpreter(PX[Interpreter]())
            uoro = yield from interpreter.getUoro
            rnnConfig = yield from interpreter.getRnnConfig
            key = yield from interpreter.updatePRNG()
            randomVector = self.distribution(key, (rnnConfig.n_h,))
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


def foldrLibrary[Interpreter, Data, Pred](
    library: Library[Interpreter, Data, Pred], param_shape: REC_PARAM
) -> Library[Interpreter, Traversable[Data], Traversable[Pred]]:
    return Library(
        model=traverseM(library.model),
        modelLossFn=accumulateM(library.modelLossFn, add, LOSS(jnp.array(0.0))),
        modelGradient=accumulateM(library.modelGradient, add, Gradient[REC_PARAM](jnp.zeros_like(param_shape))),
    )


def endowBilevelOptimization[
    OHO_Interpreter,
    OHO_Param: CanDiff,
    OHO_Label,
    TrainInterpreter: GetRecurrentParam,
    Data,
    Env,
    Param: CanDiff,
    Pred,
](
    library: Library[TrainInterpreter, Data, Pred],
    paramFn: Callable[[Gradient[Param]], MlApp[TrainInterpreter, Data, Unit]],
    trainInterpreter: TrainInterpreter,
    bilevelOptimizer: CreateLearner[OHO_Interpreter, Data, Pred],
    computeLoss: LossFn[Pred],
    resetEnvForValidation: MlApp[TrainInterpreter, Data, Unit],
) -> Library[OHO_Interpreter, Data, Pred]:
    type ST[Interpreter, Computed] = Fold[Interpreter, Data, Env, Computed]

    class _EndowBilevel_Can(
        HasInput[OHO_DATA, Data], HasPredictionInput[OHO_DATA, Data], PutLog[Env, Logs], Protocol
    ): ...

    @do()
    def newActivationStep[OHO_Interpreter: _EndowBilevel_Can]():
        @do()
        def updateParameter() -> G[MlApp[TrainInterpreter, Data, REC_PARAM]]:
            interpreter = yield from askForInterpreter(PX[TrainInterpreter]())
            return library.modelGradient.flat_map(paramFn).then(interpreter.getRecurrentParam)

        interpreter = yield from askForInterpreter(PX[OHO_Interpreter]())
        x = yield from interpreter.getInput()

        activationStep: Fold[OHO_Interpreter, OHO_DATA, Env, Param]
        activationStep = updateParameter().switch_dl(trainInterpreter).switch_data(x)
        return activationStep

    @do()
    def newPredictionStep[OHO_Interpreter: _EndowBilevel_Can]():
        interpreter = yield from askForInterpreter(PX[OHO_Interpreter]())
        x = yield from interpreter.getPredictionInput()
        env = yield from get(PX[Env]())
        predictions, _ = resetEnvForValidation.then(rnnLearner.rnn).func(trainInterpreter, x, env)
        return pure(predictions, PX3[OHO_Interpreter, OHO_DATA, Env]())

    @do()
    def newRecurrentErrorStep[OHO_Interpreter: _EndowBilevel_Can]():
        interpreter = yield from askForInterpreter(PX[OHO_Interpreter]())
        x = yield from interpreter.getPredictionInput()
        env = yield from get(PX[Env]())
        recurrentGradient, _ = resetEnvForValidation.then(rnnLearner.rnnWithGradient).func(trainInterpreter, x, env)

        validation_log = Logs(validationGradient=recurrentGradient.value)
        _ = yield from interpreter.putLog(validation_log)
        return pure(recurrentGradient, PX3[OHO_Interpreter, OHO_DATA, Env]())

    return bilevelOptimizer.createLearner(
        newActivationStep(), newPredictionStep(), computeLoss, newRecurrentErrorStep()
    )


def normalizeGradientRnnLibrary[Interpreter, Data, Env, Pred, Param: CanDiff](
    rnnLearner: RnnLibrary[Interpreter, Data, Env, Pred, Param],
) -> RnnLibrary[Interpreter, Data, Env, Pred, Param]:
    def normalizeGradient(gr: Gradient[Param]) -> Gradient[Param]:
        grad_norm = jnp.linalg.norm(gr.value) + 1e-8  # Avoid division by zero
        return Gradient(gr.value / grad_norm)

    return RnnLibrary(
        rnn=rnnLearner.rnn,
        rnnWithLoss=rnnLearner.rnnWithLoss,
        rnnWithGradient=rnnLearner.rnnWithGradient.fmap(normalizeGradient),
    )


def clipGradient[Interpreter, Data, Env, Pred, Param: CanDiff](
    clip: float,
    rnnLearner: RnnLibrary[Interpreter, Data, Env, Pred, Param],
) -> RnnLibrary[Interpreter, Data, Env, Pred, Param]:
    def clippedGradient(gr: Gradient[Param]) -> Gradient[Param]:
        clipped_gr, _ = optax.clip_by_global_norm(clip).update(gr.value, optax.EmptyState())
        return Gradient(clipped_gr)

    return RnnLibrary(
        rnn=rnnLearner.rnn,
        rnnWithLoss=rnnLearner.rnnWithLoss,
        rnnWithGradient=rnnLearner.rnnWithGradient.fmap(clippedGradient),
    )


def logGradient[Interpreter: PutLog[Env, Logs], Data, Env, Pred, Param](
    rnnLibrary: RnnLibrary[Interpreter, Data, Env, Pred, Param],
):
    type ST[Computed] = Fold[Interpreter, Data, Env, Computed]

    @do()
    def _logGradient():
        interpreter = yield from askForInterpreter(PX[Interpreter]())
        gr = yield from rnnLibrary.rnnWithGradient
        log_grad = Logs(gradient=gr.value)
        _ = yield from interpreter.putLog(log_grad)
        return pure(gr, PX3[Interpreter, Data, Env]())

    return RnnLibrary(
        rnn=rnnLibrary.rnn,
        rnnWithLoss=rnnLibrary.rnnWithLoss,
        rnnWithGradient=_logGradient(),
    )


"""
1. have oho get a HasLabel of Traversable[Label] so that it can match with Traversable[Pred]
2. oho data getinput will get a traversable[data]
3. oho data getpredictioninput will get a traversable[data], but can be of different length
then oho data is provably stackable. given a pytree, there exists a stacking algorithm that just appends +1 to front of every array. 
then unfolding this will make each elemenbt -1 dim, which gets me back to the original pytree

"""
