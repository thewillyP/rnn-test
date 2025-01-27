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

from recurrent.datarecords import OhoInputOutput
from recurrent.myfunc import flip
from recurrent.objectalgebra.typeclasses import *

from recurrent.mytypes import *
from recurrent.monad import *
from recurrent.parameters import Logs, RnnParameter, SgdParameter
from recurrent.util import pytree_split, pytreeSumZero


type LossFn[A, B] = Callable[[A, B], LOSS]
type GR[A] = Gradient[A]
type ParamFn[Dl, D, E, Pr] = Callable[[Gradient[Pr]], Fold[Dl, D, E, Unit]]


class RnnLibrary[DL, D, E, P, Pr](NamedTuple):
    rnn: Fold[DL, D, E, P]
    rnnWithLoss: Fold[DL, D, E, LOSS]
    rnnWithGradient: Fold[DL, D, E, Gradient[Pr]]


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


class _SGD[E, Pr](
    GetParameter[E, Pr],
    PutParameter[E, Pr],
    GetHyperParameter[E, SgdParameter],
    Protocol,
): ...


@do()
def doSgdStep[Dl: _SGD[E, Pr], D, E, Pr: Module](gr: GR[Pr]) -> G[Fold[Dl, D, E, Unit]]:
    dl = yield from ProxyDl[Dl].askDl()
    param = yield from dl.getParameter()
    hyperparam = yield from dl.getHyperParameter()
    new_param = jax.tree.map(
        lambda x, y: x - hyperparam.learning_rate * y, param, gr.value
    )
    return dl.putParameter(new_param)


class _R[D, E](
    GetActivation[E, ACTIVATION],
    PutActivation[E, ACTIVATION],
    GetParameter[E, RnnParameter],
    HasInput[D, Array],
    GetRnnConfig[E],
    Protocol,
): ...


@do()
def doRnnStep[Dl: _R[D, E], D, E]() -> G[Fold[Dl, D, E, ACTIVATION]]:
    dl = yield from ProxyDl[Dl].askDl()
    x = yield from dl.getInput()
    a = yield from dl.getActivation()
    param = yield from dl.getParameter()
    cfg = yield from dl.getRnnConfig()
    w_rec = jnp.reshape(param.w_rec, (cfg.n_h, cfg.n_h + cfg.n_in + 1))
    a_rec = w_rec @ jnp.concat((a, x, jnp.array([1.0])))
    a_new = ACTIVATION((1 - cfg.alpha) * a + cfg.alpha * cfg.activationFn(a_rec))
    _ = yield from dl.putActivation(a_new)
    return pure(a_new)


class _Re[E](
    GetActivation[E, ACTIVATION],
    GetParameter[E, RnnParameter],
    GetRnnConfig[E],
    Protocol,
): ...


@do()
def doRnnReadout[Dl: _Re[E], D, E]() -> G[Fold[Dl, D, E, PREDICTION]]:
    dl = yield from ProxyDl[Dl].askDl()
    a = yield from dl.getActivation()
    param = yield from dl.getParameter()
    cfg = yield from dl.getRnnConfig()
    w_out = jnp.reshape(param.w_out, (cfg.n_out, cfg.n_h + 1))
    return pure(PREDICTION(w_out @ jnp.concat((a, jnp.array([1.0])))))


class _L[D, T](HasLabel[D, T], Protocol): ...


type DoLossFn[Dl: _L[D, T], D, E, P, T] = Callable[[P], Fold[Dl, D, E, LOSS]]


def doLoss[Dl: _L[D, T], D, E, P, T](lossFn: LossFn[T, P]) -> DoLossFn[Dl, D, E, P, T]:
    @do()
    def loss_(pred: P) -> G[Fold[Dl, D, E, LOSS]]:
        d = yield from ProxyDl[Dl].askDl()
        label = yield from d.getLabel()
        loss = lossFn(pred, label)
        return pure(loss)

    return loss_


@do()
def doGradient[
    Dl, D, E, Pr: Module, T
](
    lossM: Fold[Dl, D, E, T],
    read_wrt: Fold[Dl, D, E, Pr],
    write_wrt: Callable[[Pr], Fold[Dl, D, E, Unit]],
) -> G[Fold[Dl, D, E, Gradient[Pr]]]:

    dl = yield from ProxyDl[Dl].askDl()
    d = yield from ProxyR[D].ask()
    e = yield from ProxyS[E].get()
    pr = yield from read_wrt

    def parametrized(p: Pr) -> tuple[T, E]:
        return write_wrt(p).then(lossM).func(dl, d, e)

    gr: Pr
    env: E
    gr, env = eqx.filter_jacrev(parametrized, has_aux=True)(pr)
    _ = yield from put(env)
    return pure(Gradient(gr))


@do()
def doAvgGradient[
    DL, D, E, Pr: Module
](
    step: Fold[DL, D, E, Gradient[Pr]],
    weightFn: Callable[[D], float],
    gr0: Gradient[Pr],
) -> G[Fold[DL, Traversable[D], E, Gradient[Pr]]]:

    @do()
    def doWeight(gr_accum: Gradient[Pr]) -> G[Fold[DL, D, E, Gradient[Pr]]]:
        d = yield from ProxyR[D].ask()
        gr_new = yield from step
        gr_accum_new: GR[Pr]
        gr_accum_new = jax.tree.map(
            lambda x, y: x + y * weightFn(d),
            gr_accum,
            gr_new,
        )
        return pure(gr_accum_new)

    return foldM(doWeight, gr0)


def endowAverageGradients[
    Dl: GetParameter[E, Pr], D: Module, E, P, Pr: Module
](
    rnnLibrary: RnnLibrary[Dl, Traversable[D], E, P, Pr], trunc: int, N: int
) -> RnnLibrary[Dl, Traversable[D], E, P, Pr]:
    @do()
    def avgGradient() -> G[Fold[Dl, Traversable[D], E, Gradient[Pr]]]:
        n_complete = (N // trunc) * trunc
        n_leftover = N - n_complete
        dl = yield from ProxyDl[Dl].askDl()
        pr = yield from dl.getParameter()
        ds = yield from ProxyR[Traversable[D]].ask()

        ds_scannable_, ds_leftover = pytree_split(ds, trunc)
        ds_scannable: Traversable[Traversable[D]] = Traversable(ds_scannable_)
        # ORDER MATTERS, need to scan first, then do the leftover, since they are contiguous activations
        gr_scanned = yield from doAvgGradient(
            rnnLibrary.rnnWithGradient,
            lambda _: 1.0,
            Gradient(pytreeSumZero(pr)),
        ).switch_data(ds_scannable)
        e = yield from ProxyS[E].get()

        def ifLeftover(leftover) -> Fold[Dl, Traversable[D], E, Gradient[Pr]]:
            gr_leftover, e_new = rnnLibrary.rnnWithGradient.func(dl, leftover, e)
            gr: Gradient[Pr] = jax.tree.map(
                lambda x, y: (trunc / N) * x + (n_leftover / N) * y,
                gr_scanned,
                gr_leftover,
            )
            return gr, e_new

        gr_new, e_final = jax.lax.cond(
            n_leftover == 0,
            lambda _: (jax.tree.map(lambda x: (trunc / N) * x, gr_scanned), e),
            ifLeftover,
            ds_leftover,
        )
        _ = yield from put(e_final)
        return pure(gr_new)

    return RnnLibrary(
        rnn=rnnLibrary.rnn,
        rnnWithLoss=rnnLibrary.rnnWithLoss,
        rnnWithGradient=avgGradient(),
    )


@do()
def doBatchGradients[
    Dl, D: Module | Array, E, Pr: Module
](step: Fold[Dl, D, E, Gradient[Pr]], e_dim: E) -> G[Fold[Dl, D, E, Gradient[Pr]]]:
    dl = yield from ProxyDl[Dl].askDl()
    run: Callable[[D, E], tuple[Gradient[Pr], E]]
    run = lambda d, e: step.func(dl, d, e)
    gr = yield from toFold(jax.vmap(run, in_axes=(0, e_dim), out_axes=(0, e_dim)))
    gr_summed: Pr
    gr_summed = jax.tree.map(lambda x: jnp.sum(x, axis=0), gr.value)
    return pure(Gradient(gr_summed))


@do()
def doLog[Dl: PutLog[E, Logs], D, E](loss: LOSS) -> G[Fold[Dl, D, E, LOSS]]:
    dl = yield from ProxyDl[Dl].askDl()
    _ = yield from dl.putLog(Logs(loss=loss))
    return pure(loss)


class _OfL[D, E, A, Pr, X, Y, Z](
    GetActivation[E, A],
    PutActivation[E, A],
    PutParameter[E, Pr],
    GetParameter[E, Pr],
    HasInput[D, X],
    HasLabel[D, Z],
    PutLog[E, Logs],
    Protocol,
): ...


def offlineLearning[
    Dl: _OfL[D, E, A, Pr, X, Y, Z], D, E, A, Pr, X, Y, Z, P
](
    activationStep: Fold[Dl, D, E, A],
    predictionStep: Fold[Dl, D, E, P],
    computeLoss: LossFn[Z, P],
) -> RnnLibrary[Dl, Traversable[D], E, Traversable[P], Pr]:

    rnnStep = activationStep.then(predictionStep)

    rnn = traverse(rnnStep)

    @do()
    def rnnWithLoss():
        @do()
        def accumFn(accum: LOSS) -> G[Fold[Dl, D, E, LOSS]]:
            loss = yield from rnnStep.flat_map(doLoss(computeLoss))
            return pure(LOSS(accum + loss))

        return foldM(accumFn, LOSS(jnp.array(0.0))).flat_map(doLog)

    @do()
    def rnnWithGradient():
        dl = yield from ProxyDl[Dl].askDl()
        return doGradient(rnnWithLoss(), dl.getParameter(), dl.putParameter)

    return RnnLibrary[Dl, Traversable[D], E, Traversable[P], Pr](
        rnn=rnn,
        rnnWithLoss=rnnWithLoss(),
        rnnWithGradient=rnnWithGradient(),
    )


def foldRnnLearner[
    Dl, D, E, P, Pr: Module
](rnnLearner: RnnLibrary[Dl, D, E, P, Pr], pr: Pr):
    initGr = pytreeSumZero(pr)
    return RnnLibrary[Dl, Traversable[D], E, Traversable[P], Pr](
        rnn=traverse(rnnLearner.rnn),
        rnnWithLoss=accumulate(rnnLearner.rnnWithLoss, add, LOSS(jnp.array(0.0))),
        rnnWithGradient=accumulate(
            rnnLearner.rnnWithGradient, eqx.apply_updates, Gradient(initGr)
        ),
    )


class _PfL[D, E, A, Pr, Z](
    GetActivation[E, A],
    PutActivation[E, A],
    PutParameter[E, Pr],
    GetParameter[E, Pr],
    HasLabel[D, Z],
    PutLog[E, Logs],
    Protocol,
): ...


class PastFacingLearn[Alg: _PfL[D, E, A, Pr, Z], D, E, A, Pr: Module, Z, P](ABC):

    type F[Dl, T] = Fold[Dl, D, E, T]

    @abstractmethod
    def creditAndUpdateState[
        Dl
    ](self, e_signal: F[Dl, Gradient[A]], activationStep: F[Dl, A]) -> F[
        Dl, Gradient[Pr]
    ]: ...

    class __reparam__(PutActivation[E, A], PutParameter[E, Pr], Protocol):
        pass

    def reparameterize[Dl: __reparam__](self, activationStep: F[Dl, A]):
        @do()
        def next():
            dl = yield from ProxyDl[Dl].askDl()
            d = yield from ProxyR[D].ask()
            e = yield from ProxyS[E].get()

            def parametrized(a: A, pr: Pr) -> tuple[A, E]:
                return (
                    dl.putActivation(a)
                    .then(dl.putParameter(pr))
                    .then(activationStep)
                    .func(dl, d, e)
                )

            m: Fold[Dl, D, E, Callable[[A, Pr], tuple[A, E]]]
            m = pure(parametrized)
            return m

        return next()

    def onlineLearning(
        self,
        activationStep: F[Alg, A],
        predictionStep: F[Alg, P],
        computeLoss: LossFn[Z, P],
    ):

        immedL = predictionStep.flat_map(doLoss(computeLoss)).flat_map(doLog)

        @do()
        def total_grad() -> G[Fold[Alg, D, E, Gradient[Pr]]]:
            dl = yield from ProxyDl[Alg].askDl()

            e_signal = doGradient(immedL, dl.getActivation(), dl.putActivation)

            grad_rec = yield from self.creditAndUpdateState(e_signal, activationStep)
            # order matters, need to update readout AFTER updating activation
            grad_o = yield from doGradient(immedL, dl.getParameter(), dl.putParameter)

            gradient: Gradient[Pr]
            gradient = jax.tree.map(lambda x, y: x + y, grad_o, grad_rec)
            return pure(gradient)

        return RnnLibrary[Alg, D, E, P, Pr](
            rnn=activationStep.then(predictionStep),
            rnnWithLoss=activationStep.then(immedL),
            rnnWithGradient=total_grad(),
        )


class _Infl[D, E, A, Pr, Z](
    GetInfluenceTensor[E, Gradient[Pr]],
    PutInfluenceTensor[E, Gradient[Pr]],
    _PfL[D, E, A, Pr, Z],
    Protocol,
): ...


class InfluenceTensorLearner[Alg: _Infl[D, E, A, Pr, Z], D, E, A, Pr: Module, Z, P](
    PastFacingLearn[Alg, D, E, A, Pr, Z, P], ABC
):

    type F[Dl, T] = Fold[Dl, D, E, T]
    type Grad_Pr = Gradient[Pr]

    class __infl__[E, A, Pr](
        GetInfluenceTensor[E, Gradient[Pr]],
        GetActivation[E, A],
        PutActivation[E, A],
        GetParameter[E, Pr],
        PutParameter[E, Pr],
        Protocol,
    ):
        pass

    @abstractmethod
    def influence_tensor[Dl](self, actvStep: F[Dl, A]) -> F[Dl, Grad_Pr]: ...

    def creditAndUpdateState(self, e_signal: F[Alg, Gradient[A]], actvStep: F[Alg, A]):
        @do()
        def next():
            dl = yield from ProxyDl[Alg].askDl()
            infl = yield from self.influence_tensor(actvStep)
            _ = yield from dl.putInfluenceTensor(infl)

            signal = yield from e_signal

            ca: Gradient[Pr]
            ca = jax.tree.map(lambda x: signal.value @ x, infl)
            m: Fold[Alg, D, E, Gradient[Pr]]
            m = pure(ca)

            return m

        return next()


class RTRL[Alg: _Infl[D, E, A, Pr, Z], D, E, A, Pr: Module, Z, P](
    InfluenceTensorLearner[Alg, D, E, A, Pr, Z, P]
):

    type F[Dl, T] = Fold[Dl, D, E, T]
    type Grad_Pr = Gradient[Pr]
    type constraint = InfluenceTensorLearner.__infl__[E, A, Pr]

    @do()
    def influence_tensor[Dl: constraint](self, actvStep: F[Dl, A]):
        dl = yield from ProxyDl[Dl].askDl()
        influenceTensor = yield from dl.getInfluenceTensor()
        a0 = yield from dl.getActivation()
        p0 = yield from dl.getParameter()
        parametrized = yield from self.reparameterize(actvStep)

        # I take the jvp instead of j @ v, bc I'd like to 1) do a hvp 2) forward over reverse is efficient: https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
        print(influenceTensor)
        print(a0)
        print(p0)

        immedJacInflProduct: Gradient[Pr]
        immedJacInflProduct = jax.tree.map(
            lambda x, y: jacobian_matrix_product(
                lambda a: parametrized(a, p0)[0], x, y
            ),
            a0,
            influenceTensor,
        )

        # performance below really depends on the problem. For RTRL, I should use jacrev. For OHO, jacfwd. Will make this configurable later.
        immedInfl: Pr
        env: E
        immedInfl, env = eqx.filter_jacfwd(lambda p: parametrized(a0, p), has_aux=True)(
            p0
        )

        infl: Pr
        infl = jax.tree.map(lambda x, y: x + y, immedInfl, immedJacInflProduct.value)
        _ = yield from put(env)
        m: Fold[Dl, D, E, Gradient[Pr]]
        m = pure(Gradient(infl))
        return m


class _RfL[D, E, A, Pr, Z](_Infl[D, E, A, Pr, Z], GetRfloConfig[E], Protocol): ...


class RFLO[Alg: _RfL[D, E, A, Pr, Z], D, E, A, Pr: Module, Z, P](
    InfluenceTensorLearner[Alg, D, E, A, Pr, Z, P]
):
    type F[Dl, T] = Fold[Dl, D, E, T]
    type Grad_Pr = Gradient[Pr]

    class __rflo__(
        GetRfloConfig[E], InfluenceTensorLearner.__infl__[E, A, Pr], Protocol
    ): ...

    @do()
    def influence_tensor[Dl: __rflo__](self, actvStep: F[Dl, A]):
        dl = yield from ProxyDl[Dl].askDl()
        alpha = yield from dl.getRfloConfig().fmap(lambda x: x.rflo_alpha)
        influenceTensor = yield from dl.getInfluenceTensor()
        a0 = yield from dl.getActivation()
        p0 = yield from dl.getParameter()
        parametrized = yield from self.reparameterize(actvStep)

        immedInfl: Pr
        env: E
        immedInfl, env = eqx.filter_jacrev(lambda p: parametrized(a0, p), has_aux=True)(
            p0
        )

        infl: Pr
        infl = jax.tree.map(
            lambda x, y: (1 - alpha) * x + alpha * y, influenceTensor.value, immedInfl
        )
        _ = yield from put(env)
        m: Fold[Dl, D, E, Gradient[Pr]]
        m = pure(Gradient(infl))
        return m


class _UO[D, E, A, Pr, Z](
    _PfL[D, E, A, Pr, Z],
    GetUORO[E, Pr],
    PutUORO[E, Pr],
    GetRnnConfig[E],
    HasPRNG[E, jax.Array],
    Protocol,
): ...


class UORO[Alg: _UO[D, E, A, Pr, Z], D, E, A, Pr: Module, Z, P](
    PastFacingLearn[Alg, D, E, A, Pr, Z, P]
):
    type F[Dl, T] = Fold[Dl, D, E, T]

    def __init__(self, distribution: Callable[[Array, tuple], Array]):
        super().__init__()
        self.distribution = distribution

    def gradientFlow(self, v: Array, actvStep: F[Alg, A]):
        @do()
        def next():
            dl = yield from ProxyDl[Alg].askDl()
            uoro = yield from dl.getUORO()

            a0 = yield from dl.getActivation()
            p0 = yield from dl.getParameter()
            parametrized = yield from self.reparameterize(actvStep)
            A_prev = uoro.A  # I don't distinguish between column and row tensor.

            # 1 calculate the actv_jacobian vector product with A
            immedJac_A_product: Array
            immedJac_A_product = jvp(lambda a: parametrized(a, p0)[0], a0, A_prev)

            # doing the vjp saves BIG on memory. Only use O(n^2) as we want
            fn: Callable[[Pr], tuple[A, E]] = lambda p: parametrized(a0, p)
            _, vjp_func, env = eqx.filter_vjp(fn, p0, has_aux=True)
            _ = yield from put(env)

            immedJacInfl_randomProj: Pr
            (immedJacInfl_randomProj,) = vjp_func(v)

            immed_grads: Fold[Alg, D, E, tuple[Array, Pr]]
            immed_grads = pure((immedJac_A_product, immedJacInfl_randomProj))
            return immed_grads

        return next()

    def creditAssign(self, A_: Array, B_: Gradient[Pr], e_signal: F[Alg, Gradient[A]]):
        def next(signal: Gradient[A]) -> Gradient[Pr]:
            q = signal.value @ A_
            return jax.tree.map(lambda x: x * q, B_)

        return e_signal.fmap(next)

    def creditAndUpdateState(self, e_signal: F[Alg, Gradient[A]], actvStep: F[Alg, A]):
        @do()
        def next():
            dl = yield from ProxyDl[Alg].askDl()
            uoro = yield from dl.getUORO()
            rnnConfig = yield from dl.getRnnConfig()
            #!!! WARNING, IMPURITY IN PROGRESS, will address later. Just don't run this function more than once, if autodiffing through it
            key = yield from dl.updatePRNG()
            v = self.distribution(key, (rnnConfig.n_h,))
            B_prev = uoro.B

            immedJac_A_product, immedJacInfl_randomProj = yield from self.gradientFlow(
                v, actvStep
            )

            rho0: Array = jnp.sqrt(
                pytree_norm(B_prev) / jnp.linalg.norm(immedJac_A_product)
            )
            rho1: Array = jnp.sqrt(
                pytree_norm(immedJacInfl_randomProj) / jnp.linalg.norm(v)
            )

            A_new = rho0 * immedJac_A_product + rho1 * v
            B_new: Gradient[Pr] = Gradient(
                jax.tree.map(
                    lambda x, y: x / rho0 + y / rho1,
                    B_prev.value,
                    immedJacInfl_randomProj,
                )
            )

            _ = yield from dl.putUORO(replace(uoro, A=A_new, B=B_new))

            return self.creditAssign(A_new, B_new, e_signal)

        return next()


@eqx.filter_jit
def trainStep[
    Dl, D, E
](
    learner: Fold[Dl, D, E, Unit],
    dialect: Dl,
    t_series: Traversable[D],
    initEnv: E,
) -> E:
    model = learner.func
    _, final_env = model(dialect, t_series, initEnv)
    return final_env


def endowOho[
    OHO: PastFacingLearn[OHO_Dl, D, E, Pr, OHO_Pr, Z, P],
    OHO_Dl: _PfL[D, E, Pr, OHO_Pr, Z],
    OHO_Pr: Module,
    Dl1: GetParameter[E, Pr],
    Dl2,
    D,
    E,
    P,
    Pr,
    Z,
](
    rnnLearner: RnnLibrary[Dl1 | Dl2, D, E, P, Pr],
    paramFn: ParamFn[Dl1 | Dl2, D, E, Pr],
    dl1: Dl1,
    dl2: Dl2,
    oho: OHO,
    computeLoss: LossFn[Z, P],
):
    @do()
    def parameter_step() -> G[Fold[Dl1, D, E, Pr]]:
        dl = yield from ProxyDl[Dl1].askDl()
        return rnnLearner.rnnWithGradient.flat_map(paramFn).then(dl.getParameter())

    activationStep: Fold[OHO_Dl, D, E, Pr]
    activationStep = parameter_step().switch_dl(dl1)

    predictionStep: Fold[OHO_Dl, D, E, P]
    predictionStep = rnnLearner.rnn.switch_dl(dl2)

    return oho.onlineLearning(activationStep, predictionStep, computeLoss)
