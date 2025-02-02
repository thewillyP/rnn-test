from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Callable, Protocol
from donotation import do
import jax
from jax import Array
import jax.numpy as jnp
import equinox as eqx
from equinox import Module
from operator import add

from recurrent.myfunc import compose2
from recurrent.objectalgebra.typeclasses import *

from recurrent.mytypes import *
from recurrent.monad import *
from recurrent.parameters import *
from recurrent.util import pytree_split, pytreeNumel


type LossFn[A, B] = Callable[[A, B], LOSS]
type GR[A] = Gradient[A]
type ParamFn[Dl, D, E, Pr] = Callable[[Gradient[Pr]], Fold[Dl, D, E, Unit]]


class RnnLibrary[DL, D, E, P, Pr](NamedTuple):
    rnn: Fold[DL, D, E, P]
    rnnWithLoss: Fold[DL, D, E, LOSS]
    rnnWithGradient: Fold[DL, D, E, Gradient[Pr]]


class CreateLearner[Data, Env, Actv: CanDiff, Param: CanDiff, Label, Pred](Protocol):
    type ST[Interpreter, Computed] = Fold[Interpreter, Data, Env, Computed]

    def createLearner[Interpreter](
        self,
        activationStep: ST[Interpreter, Actv],
        predictionStep: ST[Interpreter, Param],
        lossFunction: LossFn[Label, Pred],
    ) -> RnnLibrary[Interpreter, Data, Env, Pred, Param]: ...


@eqx.filter_jit
def pytree_norm(tree):
    squared = jax.tree.map(lambda x: jnp.sum(x**2), tree)
    return jnp.sqrt(jax.tree.reduce(jnp.add, squared))


@eqx.filter_jit
def jvp(f, primal, tangent):
    return eqx.filter_jvp(f, (primal,), (tangent,))[1]


@eqx.filter_jit
def jacobian_matrix_product(f, primal, matrix):
    wrapper = lambda p, t: jvp(f, p, t)
    return jax.vmap(wrapper, in_axes=(None, 1), out_axes=1)(primal, matrix)


@do()
def resetRnnActivation[Interpreter: PutActivation[Env, Actv], Data, Env, Actv](
    resetActv: Actv,
) -> G[Fold[Interpreter, Data, Env, Unit]]:
    interpreter = yield from askForInterpreter(PX[Interpreter]())
    return interpreter.putActivation(resetActv)


class _SGD_Can[Env, Param](
    GetParameter[Env, Param],
    PutParameter[Env, Param],
    GetHyperParameter[Env, SgdParameter],
    Protocol,
): ...


@do()
def doSgdStep[Interpreter: _SGD_Can[Env, Param], Data, Env, Param: CanDiff](
    gr: Gradient[Param],
) -> G[Fold[Interpreter, Data, Env, Unit]]:
    interpreter = yield from askForInterpreter(PX[Interpreter]())
    isParam = yield from interpreter.getParameter()
    hyperparam = yield from interpreter.getHyperParameter()
    new_param = invmap(isParam, lambda x: jnp.ravel(x - hyperparam.learning_rate * gr.value))
    return interpreter.putParameter(new_param)


class _RnnActivation_Can[Data, Env](
    GetActivation[Env, ACTIVATION],
    PutActivation[Env, ACTIVATION],
    GetParameter[Env, RnnParameter],
    HasInput[Data, Array],
    GetRnnConfig[Env],
    Protocol,
): ...


@do()
def doRnnStep[Interpreter: _RnnActivation_Can[Data, Env], Data, Env]():
    interpreter = yield from askForInterpreter(PX[Interpreter]())
    x = yield from interpreter.getInput()
    a = yield from interpreter.getActivation()
    param = yield from interpreter.getParameter()
    cfg = yield from interpreter.getRnnConfig()

    a_rec = param.w_rec @ jnp.concat((a, x, jnp.asarray([1.0])))
    a_new = ACTIVATION((1 - cfg.alpha) * a + cfg.alpha * cfg.activationFn(a_rec))
    _ = yield from interpreter.putActivation(a_new)
    return pure(a_new, PX3[Interpreter, Data, Env]())


class _RnnReadout_Can[Env](
    GetActivation[Env, ACTIVATION],
    GetParameter[Env, RnnParameter],
    GetRnnConfig[Env],
    Protocol,
): ...


@do()
def doRnnReadout[Interpreter: _RnnReadout_Can[Env], Data, Env]():
    interpreter = yield from askForInterpreter(PX[Interpreter]())
    a = yield from interpreter.getActivation()
    param = yield from interpreter.getParameter()
    pred = PREDICTION(param.w_out @ jnp.concat((a, jnp.asarray([1.0]))))
    return pure(pred, PX3[Interpreter, Data, Env]())


def doLoss[Data, Env, Pred, Label](lossFn: LossFn[Label, Pred]):
    class _Loss_Can(HasLabel[Data, Label], Protocol): ...

    @do()
    def _lossFn[Interpreter: _Loss_Can](pred: Pred):
        interpreter = yield from askForInterpreter(PX[Interpreter]())
        label = yield from interpreter.getLabel()
        loss = lossFn(pred, label)
        return pure(loss, PX3[Interpreter, Data, Env]())

    return _lossFn


type FnFmap[Inp, A, B] = Callable[[Callable[[Inp], A]], Callable[[Inp], B]]


@do()
def doDifferentiation[Interpreter, Data, Env, Param: CanDiff, T: CanDiff](
    model: Fold[Interpreter, Data, Env, T],
    read_wrt: Fold[Interpreter, Data, Env, Param],
    write_wrt: Callable[[Param], Fold[Interpreter, Data, Env, Unit]],
    jax_diff: FnFmap[Array, tuple[Array, Env], tuple[Array, Env]],
) -> G[Fold[Interpreter, Data, Env, jax.Array]]:
    print("recompiled")
    interpreter = yield from askForInterpreter(PX[Interpreter]())
    data = yield from ask(PX[Data]())
    env = yield from get(PX[Env]())
    param = yield from read_wrt
    param_vector = toVector(endowVector(param))

    def parametrized(p: Array):
        p_ = invmap(param, lambda _: p)
        return write_wrt(p_).then(model).fmap(lambda t: toVector(endowVector(t))).func(interpreter, data, env)

    grad, new_env = jax_diff(parametrized)(param_vector)
    # grad will always be atleast2d since toVector always returns atleast1d, even for loss

    _ = yield from put(new_env)
    return pure(grad, PX3[Interpreter, Data, Env]())


# no auto curry so bulk in code is unavoidable
def doJacobian[Interpreter, Data, Env, Param: CanDiff, T: CanDiff](
    model: Fold[Interpreter, Data, Env, T],
    read_wrt: Fold[Interpreter, Data, Env, Param],
    write_wrt: Callable[[Param], Fold[Interpreter, Data, Env, Unit]],
):
    return doDifferentiation(model, read_wrt, write_wrt, lambda f: eqx.filter_jacrev(f, has_aux=True)).fmap(
        Jacobian[Param]
    )


def doGradient[Interpreter, Data, Env, Param: CanDiff](
    model: Fold[Interpreter, Data, Env, jax.Array],
    read_wrt: Fold[Interpreter, Data, Env, Param],
    write_wrt: Callable[[Param], Fold[Interpreter, Data, Env, Unit]],
):
    return doDifferentiation(model, read_wrt, write_wrt, lambda f: eqx.filter_jacrev(f, has_aux=True)).fmap(
        lambda x: Gradient[Param](jnp.ravel(x))  # enforce invariant, since doDiff always returns atleast2d
    )


def endowAveragedGradients[Interpreter, Data: Module, Env, Pred, Param: CanDiff](
    rnnLibrary: RnnLibrary[Interpreter, Traversable[Data], Env, Pred, Param], trunc: int, N: int, param_shape: Param
) -> RnnLibrary[Interpreter, Traversable[Data], Env, Pred, Param]:
    type ST[Computed] = Fold[Interpreter, Traversable[Data], Env, Computed]

    @do()
    def avgGradient() -> G[ST[Gradient[Param]]]:
        n_complete = (N // trunc) * trunc
        n_leftover = N - n_complete
        dataset = yield from ask(PX[Traversable[Data]]())

        # Reshaping the dataset with 'trunc' makes scanning easier IF dataset is evenly shaped,
        # but JAX doesn’t allow jagged arrays. So, we adjust the data
        # to be evenly shaped for (scan . scan) and handle the rest separately.
        # ORDER MATTERS, need to scan first, then do the leftover, since they are contiguous activations
        ds_scannable_, ds_leftover = pytree_split(dataset, trunc)
        ds_scannable: Traversable[Traversable[Data]] = Traversable(ds_scannable_)

        gr_scanned = yield from accumulate(
            rnnLibrary.rnnWithGradient, add, Gradient[Param](jnp.zeros(pytreeNumel(param_shape)))
        ).switch_data(ds_scannable)

        interpreter = yield from askForInterpreter(PX[Interpreter]())
        env_after_scannable = yield from get(PX[Env]())

        def ifLeftoverData(leftover) -> ST[Gradient[Param]]:
            gr_leftover, e_new = rnnLibrary.rnnWithGradient.func(interpreter, leftover, env_after_scannable)
            return Gradient((trunc / N) * gr_scanned.value + (n_leftover / N) * gr_leftover.value), e_new

        gradient_final, env_final = jax.lax.cond(
            n_leftover == 0,
            lambda _: (Gradient((trunc / N) * gr_scanned.value), env_after_scannable),
            ifLeftoverData,
            ds_leftover,
        )
        _ = yield from put(env_final)
        return pure(gradient_final, PX3[Interpreter, Traversable[Data], Env]())

    return RnnLibrary(
        rnn=rnnLibrary.rnn,
        rnnWithLoss=rnnLibrary.rnnWithLoss,
        rnnWithGradient=avgGradient(),
    )


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


class OfflineLearning[Data, Env, Actv: CanDiff, Param: CanDiff, Label, Pred](ABC):
    class _OfflineLearner_Can(
        PutParameter[Env, Param],
        GetParameter[Env, Param],
        HasLabel[Data, Label],
        Protocol,
    ): ...

    type ST[Interpreter, Computed] = Fold[Interpreter, Data, Env, Computed]

    def createLearner[Interpreter: _OfflineLearner_Can](
        self,
        activationStep: ST[Interpreter, Actv],
        predictionStep: ST[Interpreter, Param],
        lossFunction: LossFn[Label, Pred],
    ):
        rnnStep = activationStep.then(predictionStep)
        rnn = traverse(rnnStep)
        rnnWithLoss = accumulate(rnnStep.flat_map(doLoss(lossFunction)), add, LOSS(jnp.Array(0.0)))

        @do()
        def rnnWithGradient():
            interpreter: Interpreter = yield from askForInterpreter(PX[Interpreter]())
            return doGradient(rnnWithLoss, interpreter.getParameter(), interpreter.putParameter)

        return RnnLibrary[Interpreter, Traversable[Data], Env, Traversable[Pred], Param](
            rnn=rnn,
            rnnWithLoss=rnnWithLoss,
            rnnWithGradient=rnnWithGradient(),
        )


class PastFacingLearn[Data, Env, Actv: CanDiff, Param: CanDiff, Label, Pred](ABC):
    class _PastFacingLearner_Can(
        GetActivation[Env, Actv],
        PutActivation[Env, Actv],
        PutParameter[Env, Param],
        GetParameter[Env, Param],
        HasLabel[Data, Label],
        Protocol,
    ): ...

    type ST[Interpreter, Computed] = Fold[Interpreter, Data, Env, Computed]

    @abstractmethod
    def creditAssignment[Interpreter](
        self,
        recurrentError: ST[Interpreter, Gradient[Actv]],
        activationStep: ST[Interpreter, Actv],
    ) -> ST[Interpreter, Gradient[Param]]: ...

    class _CreateModel_Can(
        GetActivation[Env, Actv], GetParameter[Env, Param], PutActivation[Env, Actv], PutParameter[Env, Param], Protocol
    ): ...

    @staticmethod
    def createRnnForward[Interpreter: _CreateModel_Can](
        activationStep: ST[Interpreter, Actv],
    ):
        @do()
        def _creatingModel():
            interpreter = yield from askForInterpreter(PX[Interpreter]())
            data = yield from ask(PX[Data]())
            env = yield from get(PX[Env]())
            activation = yield from interpreter.getActivation()
            parameter = yield from interpreter.getParameter()

            def parametrized(actv_vec: Array, param_vec: Array):
                actv_ = invmap(activation, lambda _: actv_vec)
                param_ = invmap(parameter, lambda _: param_vec)
                return (
                    interpreter.putActivation(actv_)
                    .then(interpreter.putParameter(param_))
                    .then(activationStep)
                    .fmap(lambda t: toVector(endowVector(t)))
                    .func(interpreter, data, env)
                )

            return pure(parametrized, PX3[Interpreter, Data, Env]())

        return _creatingModel()

    def createLearner[Interpreter: _PastFacingLearner_Can](
        self,
        activationStep: ST[Interpreter, Actv],
        predictionStep: ST[Interpreter, Param],
        lossFunction: LossFn[Label, Pred],
    ):
        willGetImmediateLoss = predictionStep.flat_map(doLoss(lossFunction))

        @do()
        def rnnGradient() -> G[Fold[Interpreter, Data, Env, Gradient[Param]]]:
            interpreter = yield from askForInterpreter(PX[Interpreter]())
            willGetRecurrentError = doGradient(
                willGetImmediateLoss,
                interpreter.getActivation(),
                interpreter.putActivation,
            )

            # order matters, need to update readout (grad_o) AFTER updating activation (grad_rec updates it)
            grad_rec = yield from self.creditAssignment(willGetRecurrentError, activationStep)
            grad_readout = yield from doGradient(
                willGetImmediateLoss,
                interpreter.getParameter(),
                interpreter.putParameter,
            )

            return pure(grad_rec + grad_readout, PX3[Interpreter, Data, Env]())

        return RnnLibrary[Interpreter, Data, Env, Pred, Param](
            rnn=activationStep.then(predictionStep),
            rnnWithLoss=activationStep.then(willGetImmediateLoss),
            rnnWithGradient=rnnGradient(),
        )


class InfluenceTensorLearner[Data, Env, Actv: CanDiff, Param: CanDiff, Label, Pred](
    PastFacingLearn[Data, Env, Actv, Param, Label, Pred]
):
    class _InfluenceTensorLearner_Can(
        GetInfluenceTensor[Env, Jacobian[Param]],
        PutInfluenceTensor[Env, Jacobian[Param]],
        PastFacingLearn[Data, Env, Actv, Param, Label, Pred]._PastFacingLearner_Can,
        Protocol,
    ): ...

    type ST[Interpreter, Computed] = Fold[Interpreter, Data, Env, Computed]

    @abstractmethod
    def getInfluenceTensor[Interpreter](
        self, activationStep: ST[Interpreter, Actv]
    ) -> ST[Interpreter, Jacobian[Param]]: ...

    def creditAssignment[Interpreter: _InfluenceTensorLearner_Can](
        self, recurrentError: ST[Interpreter, Gradient[Actv]], activationStep: ST[Interpreter, Actv]
    ):
        @do()
        def _creditAssignment():
            interpreter = yield from askForInterpreter(PX[Interpreter]())
            influenceTensor = yield from self.getInfluenceTensor(activationStep)
            signal = yield from recurrentError
            recurrentGradient = Gradient[Param](signal.value @ influenceTensor.value)
            _ = yield from interpreter.putInfluenceTensor(influenceTensor)
            return pure(recurrentGradient, PX3[Interpreter, Data, Env]())

        return _creditAssignment()


class RTRL[Data, Env, Actv: CanDiff, Param: CanDiff, Label, Pred](
    InfluenceTensorLearner[Data, Env, Actv, Param, Label, Pred]
):
    type ST[Interpreter, Computed] = Fold[Interpreter, Data, Env, Computed]

    class _UpdateInfluence_Can(
        GetInfluenceTensor[Env, Jacobian[Param]],
        GetActivation[Env, Actv],
        PutActivation[Env, Actv],
        GetParameter[Env, Param],
        PutParameter[Env, Param],
        Protocol,
    ): ...

    @do()
    def getInfluenceTensor[Interpreter: _UpdateInfluence_Can, Data, Env](self, activationStep: ST[Interpreter, Actv]):
        interpreter = yield from askForInterpreter(PX[Interpreter]())
        rnnForward = yield from self.createRnnForward(activationStep)

        influenceTensor = yield from interpreter.getInfluenceTensor()
        actv0 = yield from interpreter.getActivation().fmap(compose2(endowVector, toVector))
        param0 = yield from interpreter.getParameter().fmap(compose2(endowVector, toVector))

        # I take the jvp instead of j @ v, bc I'd like to 1) do a hvp 2) forward over reverse is efficient: https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
        immediateJacobian__InfluenceTensor_product: Array = jacobian_matrix_product(
            lambda a: rnnForward(a, param0)[0],
            actv0,
            influenceTensor.value,
        )
        immediateInfluence: Array
        env: Env
        immediateInfluence, env = eqx.filter_jacfwd(lambda p: rnnForward(actv0, p), has_aux=True)(param0)

        newInfluenceTensor: Jacobian[Param] = Jacobian(immediateJacobian__InfluenceTensor_product + immediateInfluence)

        _ = yield from put(env)
        return pure(newInfluenceTensor, PX3[Interpreter, Data, Env]())


class RFLO[Data, Env, Actv: CanDiff, Param: CanDiff, Label, Pred](
    InfluenceTensorLearner[Data, Env, Actv, Param, Label, Pred]
):
    type ST[Interpreter, Computed] = Fold[Interpreter, Data, Env, Computed]

    class _UpdateRFLO_Can(
        GetRnnConfig[Env],
        GetInfluenceTensor[Env, Jacobian[Param]],
        GetActivation[Env, Actv],
        PutActivation[Env, Actv],
        GetParameter[Env, Param],
        PutParameter[Env, Param],
        Protocol,
    ): ...

    @do()
    def getInfluenceTensor[Interpreter: _UpdateRFLO_Can, Data, Env](self, activationStep: ST[Interpreter, Actv]):
        interpreter = yield from askForInterpreter(PX[Interpreter]())
        rnnForward = yield from self.createRnnForward(activationStep)

        alpha = yield from interpreter.getRnnConfig().fmap(lambda x: x.alpha)
        influenceTensor = yield from interpreter.getInfluenceTensor()
        actv0 = yield from interpreter.getActivation().fmap(compose2(endowVector, toVector))
        param0 = yield from interpreter.getParameter().fmap(compose2(endowVector, toVector))

        immediateInfluence: Array
        env: Env
        immediateInfluence, env = eqx.filter_jacrev(lambda p: rnnForward(actv0, p), has_aux=True)(param0)

        newInfluenceTensor: Jacobian[Param] = Jacobian((1 - alpha) * influenceTensor.value + alpha * immediateInfluence)

        _ = yield from put(env)
        return pure(newInfluenceTensor, PX3[Interpreter, Data, Env]())


class UORO[Data, Env, Actv: Module, Param: Module, Label, Pred](PastFacingLearn[Data, Env, Actv, Param, Label, Pred]):
    type ST[Interpreter, Computed] = Fold[Interpreter, Data, Env, Computed]

    class _UORO_Can(
        PastFacingLearn[Data, Env, Actv, Param, Label, Pred]._PastFacingLearner_Can,
        GetUORO[Env],
        PutUORO[Env],
        GetRnnConfig[Env],
        HasPRNG[Env, jax.Array],
        Protocol,
    ): ...

    def __init__(self, distribution: Callable[[Array, tuple[int]], Array]):
        self.distribution = distribution

    def getProjectedGradients[Interpreter: _UORO_Can](self, randomVector: Array, activationStep: ST[Interpreter, Actv]):
        @do()
        def projectGradients():
            interpreter = yield from askForInterpreter(PX[Interpreter]())
            rnnForward = yield from self.createRnnForward(activationStep)

            uoro = yield from interpreter.getUORO()
            actv0 = yield from interpreter.getActivation().fmap(compose2(endowVector, toVector))
            param0 = yield from interpreter.getParameter().fmap(compose2(endowVector, toVector))

            # 1 calculate the actv_jacobian vector product with A
            immediateJacobian__A_projection: Array
            immediateJacobian__A_projection = jvp(
                lambda a: rnnForward(a, param0)[0],
                actv0,
                uoro.A,
            )

            # 2 doing the vjp saves BIG on memory. Only use O(n^2) as we want
            fn: Callable[[Array], tuple[Array, Env]] = lambda p: rnnForward(actv0, p)
            _, vjp_func, env = eqx.filter_vjp(fn, param0, has_aux=True)
            immediateInfluence__Random_projection: Array
            (immediateInfluence__Random_projection,) = vjp_func(randomVector)

            assert isinstance(immediateJacobian__A_projection, Array)
            assert isinstance(immediateInfluence__Random_projection, Array)

            _ = yield from put(env)
            return pure(
                (immediateJacobian__A_projection, immediateInfluence__Random_projection), PX3[Interpreter, Data, Env]()
            )

        return projectGradients()

    def propagateRecurrentError[Interpreter: _UORO_Can](
        self, A_: Array, B_: Array, recurrentError: ST[Interpreter, Gradient[Actv]]
    ):
        def propagate(signal: Gradient[Actv]) -> Gradient[Param]:
            q = signal.value @ A_
            return Gradient(q * B_)

        return recurrentError.fmap(propagate)

    def creditAssignment[Interpreter: _UORO_Can](
        self, recurrentError: ST[Interpreter, Gradient[Actv]], activationStep: ST[Interpreter, Actv]
    ):
        @do()
        def creditAssignment_():
            interpreter = yield from askForInterpreter(PX[Interpreter]())
            uoro = yield from interpreter.getUORO()
            rnnConfig = yield from interpreter.getRnnConfig()
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
            _ = yield from interpreter.putUORO(replace(uoro, A=A_new, B=B_new))

            return self.propagateRecurrentError(A_new, B_new, recurrentError)

        return creditAssignment_()


def foldrRnnLearner[Interpreter, Data, Env, Pred, Param: Module](
    rnnLearner: RnnLibrary[Interpreter, Data, Env, Pred, Param], param_shape: Param
):
    return RnnLibrary[Interpreter, Traversable[Data], Env, Traversable[Pred], Param](
        rnn=traverse(rnnLearner.rnn),
        rnnWithLoss=accumulate(rnnLearner.rnnWithLoss, add, LOSS(jnp.array(0.0))),
        rnnWithGradient=accumulate(
            rnnLearner.rnnWithGradient, add, Gradient[Param](jnp.zeros(pytreeNumel(param_shape)))
        ),
    )


@eqx.filter_jit
def trainStep[Dl, D, E](
    learner: Fold[Dl, D, E, Unit],
    dialect: Dl,
    t_series: Traversable[D],
    initEnv: E,
) -> E:
    model = learner.func
    _, final_env = model(dialect, t_series, initEnv)
    return final_env


def endowBilevelOptimization[
    OHO_Interpreter,
    OHO_Param: CanDiff,
    OHO_Label,
    OHO_DATA,
    TrainInterpreter: GetParameter[Env, Param],
    Data,
    Env,
    Param: CanDiff,
    Pred,
](
    rnnLearner: RnnLibrary[TrainInterpreter, Data, Env, Pred, Param],
    paramFn: Callable[[Gradient[Param]], Fold[TrainInterpreter, Data, Env, Unit]],
    trainInterpreter: TrainInterpreter,
    bilevelOptimizer: CreateLearner[OHO_DATA, Env, Param, OHO_Param, OHO_Label, Pred],
    computeLoss: LossFn[OHO_Label, Pred],
    resetEnvForValidation: Fold[TrainInterpreter, Data, Env, Unit],
) -> RnnLibrary[OHO_Interpreter, OHO_DATA, Env, Pred, OHO_Param]:
    type ST[Interpreter, Computed] = Fold[Interpreter, Data, Env, Computed]

    class _EndowBilevel_Can(HasInput[OHO_DATA, Data], HasPredictionInput[OHO_DATA, Data], Protocol): ...

    @do()
    def newActivationStep[OHO_Interpreter: _EndowBilevel_Can]():
        @do()
        def updateParameter() -> G[ST[TrainInterpreter, Param]]:
            interpreter = yield from askForInterpreter(PX[TrainInterpreter]())
            return rnnLearner.rnnWithGradient.flat_map(paramFn).then(interpreter.getParameter())

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

    return bilevelOptimizer.createLearner(newActivationStep(), newPredictionStep(), computeLoss)


"""
1. have oho get a HasLabel of Traversable[Label] so that it can match with Traversable[Pred]
2. oho data getinput will get a traversable[data]
3. oho data getpredictioninput will get a traversable[data], but can be of different length
then oho data is provably stackable. given a pytree, there exists a stacking algorithm that just appends +1 to front of every array. 
then unfolding this will make each elemenbt -1 dim, which gets me back to the original pytree

"""
