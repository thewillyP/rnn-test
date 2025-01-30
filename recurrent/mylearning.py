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
from recurrent.myfunc import curry, flip
from recurrent.objectalgebra.typeclasses import *

from recurrent.mytypes import *
from recurrent.monad import *
from recurrent.parameters import *
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
    GetParameter[E, IsVector[Pr]],
    PutParameter[E, IsVector[Pr]],
    GetHyperParameter[E, IsVector[SgdParameter]],
    Protocol,
): ...


@do()
def doSgdStep[Dl: _SGD[E, Pr], D, E, Pr: Module](gr: GR[Pr]) -> G[Fold[Dl, D, E, Unit]]:
    dl = yield from ProxyDl[Dl].askDl()
    isParam = yield from dl.getParameter()
    hyperparam = yield from dl.getHyperParameter().fmap(toParam)
    new_param = invmap(isParam, lambda x: x - hyperparam.learning_rate * gr.value)
    return dl.putParameter(new_param)


class _R[D, E](
    GetActivation[E, IsVector[ACTIVATION]],
    PutActivation[E, IsVector[ACTIVATION]],
    GetParameter[E, IsVector[RnnParameter]],
    HasInput[D, Array],
    GetRnnConfig[E],
    Protocol,
): ...


@do()
def doRnnStep[Dl: _R[D, E], D, E]() -> G[Fold[Dl, D, E, IsVector[ACTIVATION]]]:
    dl = yield from ProxyDl[Dl].askDl()
    x = yield from dl.getInput()
    activ_v = yield from dl.getActivation()
    a = toVector(activ_v)
    param = yield from dl.getParameter().fmap(toParam)
    cfg = yield from dl.getRnnConfig()
    a_rec = param.w_rec @ jnp.concat((a, x, jnp.array([1.0])))
    a_new = ACTIVATION((1 - cfg.alpha) * a + cfg.alpha * cfg.activationFn(a_rec))
    new_activ_v = invmap(activ_v, lambda _: a_new)
    _ = yield from dl.putActivation(new_activ_v)
    return pure(new_activ_v)


class _Re[E](
    GetActivation[E, IsVector[ACTIVATION]],
    GetParameter[E, IsVector[RnnParameter]],
    GetRnnConfig[E],
    Protocol,
): ...


@do()
def doRnnReadout[Dl: _Re[E], D, E]() -> G[Fold[Dl, D, E, PREDICTION]]:
    dl = yield from ProxyDl[Dl].askDl()
    a = yield from dl.getActivation().fmap(toVector)
    param = yield from dl.getParameter().fmap(toParam)
    return pure(PREDICTION(param.w_out @ jnp.concat((a, jnp.array([1.0])))))


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


type FnFmap[Inp, A, B] = Callable[[Callable[[Inp], A]], Callable[[Inp], B]]


@do()
def fold_fmap[
    Dl, D, E, Inp, A, B
](
    action: Fold[Dl, D, E, A],
    read_wrt: Fold[Dl, D, E, Inp],
    write_wrt: Callable[[Inp], Fold[Dl, D, E, Unit]],
    fn_fmap: FnFmap[Inp, tuple[A, E], B],
) -> G[Fold[Dl, D, E, B]]:

    dl = yield from ProxyDl[Dl].askDl()
    d = yield from ProxyR[D].ask()
    e = yield from ProxyS[E].get()
    inp = yield from read_wrt

    def parametrized(i: Inp) -> tuple[A, E]:
        return write_wrt(i).then(action).func(dl, d, e)

    b = fn_fmap(parametrized)(inp)
    return pure(b)


@do()
def doDifferentiation[
    Dl, D, E, Pr: Module, A: Module, T: IsVector[A] | Array
](
    model: Fold[Dl, D, E, T],
    read_wrt: Fold[Dl, D, E, IsVector[Pr]],
    write_wrt: Callable[[IsVector[Pr]], Fold[Dl, D, E, Unit]],
    jax_diff: FnFmap[IsVector[Pr], tuple[T, E], tuple[IsVector[Pr], E]],
) -> G[Fold[Dl, D, E, jax.Array]]:
    gr, env = yield from fold_fmap(
        model,
        read_wrt,
        write_wrt,
        jax_diff,
    )
    _ = yield from put(env)
    return pure(toVector(gr))


# no auto curry so bulk in code is unavoidable
def doJacobian[
    Dl, D, E, Pr: Module, T
](
    model: Fold[Dl, D, E, IsVector[T]],
    read_wrt: Fold[Dl, D, E, IsVector[Pr]],
    write_wrt: Callable[[IsVector[Pr]], Fold[Dl, D, E, Unit]],
):
    return doDifferentiation(model, read_wrt, write_wrt, lambda f: eqx.filter_jacrev(f, has_aux=True)).fmap(
        Jacobian[Pr]
    )


def doGradient[
    Dl, D, E, Pr: Module
](
    model: Fold[Dl, D, E, jax.Array],
    read_wrt: Fold[Dl, D, E, IsVector[Pr]],
    write_wrt: Callable[[IsVector[Pr]], Fold[Dl, D, E, Unit]],
):
    return doDifferentiation(model, read_wrt, write_wrt, lambda f: eqx.filter_jacrev(f, has_aux=True)).fmap(
        Gradient[Pr]
    )


@do()
def doAvgGradient[
    DL, D, E, Pr: Module
](step: Fold[DL, D, E, Gradient[Pr]], weightFn: Callable[[D], float],) -> G[Fold[DL, Traversable[D], E, Gradient[Pr]]]:
    @do()
    def doWeight() -> G[Fold[DL, D, E, Gradient[Pr]]]:
        d = yield from ProxyR[D].ask()
        gr_new = yield from step
        weighted = gr_new.value * weightFn(d)
        return pure(Gradient(weighted))

    return accumulate(doWeight(), add, Gradient[Pr](0.0))


def endowAverageGradients[
    Dl, D: Module, E, P, Pr: Module
](rnnLibrary: RnnLibrary[Dl, Traversable[D], E, P, Pr], trunc: int, N: int) -> RnnLibrary[Dl, Traversable[D], E, P, Pr]:
    @do()
    def avgGradient() -> G[Fold[Dl, Traversable[D], E, Gradient[Pr]]]:
        n_complete = (N // trunc) * trunc
        n_leftover = N - n_complete
        dl = yield from ProxyDl[Dl].askDl()
        ds = yield from ProxyR[Traversable[D]].ask()

        ds_scannable_, ds_leftover = pytree_split(ds, trunc)
        ds_scannable: Traversable[Traversable[D]] = Traversable(ds_scannable_)
        # ORDER MATTERS, need to scan first, then do the leftover, since they are contiguous activations
        gr_scanned: Gradient[Pr] = yield from doAvgGradient(
            rnnLibrary.rnnWithGradient,
            lambda _: 1.0,
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


# todo, recheck later
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


class _OfL[D, E, Pr, Label](
    PutParameter[E, IsVector[Pr]],
    GetParameter[E, IsVector[Pr]],
    HasLabel[D, Label],
    PutLog[E, Logs],
    Protocol,
): ...


def offlineLearning[
    Dl: _OfL[D, E, Pr, Label], D, E, A, Pr: Module, Label, Pred
](
    activationStep: Fold[Dl, D, E, A],
    predictionStep: Fold[Dl, D, E, Pred],
    computeLoss: LossFn[Label, Pred],
) -> RnnLibrary[Dl, Traversable[D], E, Traversable[Pred], Pr]:

    rnnStep = activationStep.then(predictionStep)
    rnn = traverse(rnnStep)
    rnnWithLoss = accumulate(rnnStep.flat_map(doLoss(computeLoss)), add, LOSS(0.0))

    @do()
    def rnnWithGradient():
        dl = yield from ProxyDl[Dl].askDl()
        return doGradient(rnnWithLoss, dl.getParameter(), dl.putParameter)

    return RnnLibrary(
        rnn=rnn,
        rnnWithLoss=rnnWithLoss,
        rnnWithGradient=rnnWithGradient(),
    )


def foldRnnLearner[Dl, D, E, P, Pr: Module](rnnLearner: RnnLibrary[Dl, D, E, P, Pr]):
    return RnnLibrary[Dl, Traversable[D], E, Traversable[P], Pr](
        rnn=traverse(rnnLearner.rnn),
        rnnWithLoss=accumulate(rnnLearner.rnnWithLoss, add, LOSS(jnp.array(0.0))),
        rnnWithGradient=accumulate(rnnLearner.rnnWithGradient, add, Gradient[Pr](0.0)),
    )


class _PfL[D, E, A, Pr, Label](
    GetActivation[E, IsVector[A]],
    PutActivation[E, IsVector[A]],
    PutParameter[E, IsVector[Pr]],
    GetParameter[E, IsVector[Pr]],
    HasLabel[D, Label],
    PutLog[E, Logs],
    Protocol,
): ...


class PastFacingLearn[Alg: _PfL[D, E, A, Pr, Lbl], D, E, A: Module, Pr: Module, Lbl, P](ABC):

    type F[Dl, T] = Fold[Dl, D, E, T]

    @abstractmethod
    def recGrad[Dl](s, sig: F[Dl, GR[A]], actv: F[Dl, IsVector[A]]) -> F[Dl, GR[Pr]]:
        pass

    class __reparam__(PutActivation[E, A], PutParameter[E, Pr], Protocol): ...

    @staticmethod
    def reparameterize[Dl: __reparam__](activationStep: F[Dl, IsVector[A]]):

        @do()
        def next():
            dl = yield from ProxyDl[Dl].askDl()
            d = yield from ProxyR[D].ask()
            e = yield from ProxyS[E].get()

            def parametrized(a: IsVector[A], pr: IsVector[Pr]):
                return dl.putActivation(a).then(dl.putParameter(pr)).then(activationStep).func(dl, d, e)

            m: Fold[Dl, D, E, Callable[[IsVector[A], IsVector[Pr]], tuple[IsVector[A], E]]]
            m = pure(parametrized)
            return m

        return next()

    def onlineLearning(
        self,
        activationStep: F[Alg, IsVector[A]],
        predictionStep: F[Alg, IsVector[Pr]],
        computeLoss: LossFn[Lbl, P],
    ):

        immedL = predictionStep.flat_map(doLoss(computeLoss)).flat_map(doLog)

        @do()
        def total_grad() -> G[Fold[Alg, D, E, Gradient[Pr]]]:
            dl = yield from ProxyDl[Alg].askDl()
            e_signal = doGradient(immedL, dl.getActivation(), dl.putActivation)

            # order matters, need to update readout (grad_o) AFTER updating activation (grad_rec updates it)
            grad_rec = yield from self.recGrad(e_signal, activationStep)
            grad_o = yield from doGradient(immedL, dl.getParameter(), dl.putParameter)
            return pure(grad_rec + grad_o)

        return RnnLibrary[Alg, D, E, P, Pr](
            rnn=activationStep.then(predictionStep),
            rnnWithLoss=activationStep.then(immedL),
            rnnWithGradient=total_grad(),
        )


class _Infl[D, E, A, Pr, Label](
    GetInfluenceTensor[E, Jacobian[Pr]],
    PutInfluenceTensor[E, Jacobian[Pr]],
    _PfL[D, E, A, Pr, Label],
    Protocol,
): ...


class InfluenceTensorLearner[Alg: _Infl[D, E, A, Pr, Lbl], D, E, A: Module, Pr: Module, Lbl, P](
    PastFacingLearn[Alg, D, E, A, Pr, Lbl, P], ABC
):

    type F[Dl, T] = Fold[Dl, D, E, T]

    class __infl__[E, A, Pr](
        GetInfluenceTensor[E, Jacobian[Pr]],
        GetActivation[E, IsVector[A]],
        PutActivation[E, IsVector[A]],
        GetParameter[E, IsVector[Pr]],
        PutParameter[E, IsVector[Pr]],
        Protocol,
    ):
        pass

    @abstractmethod
    def influence_tensor[Dl](self, actv: F[Dl, IsVector[A]]) -> F[Dl, Jacobian[Pr]]: ...

    def recGrad(self, e_signal: F[Alg, Gradient[A]], actvStep: F[Alg, IsVector[A]]):
        @do()
        def next():
            dl = yield from ProxyDl[Alg].askDl()
            infl = yield from self.influence_tensor(actvStep)
            _ = yield from dl.putInfluenceTensor(infl)

            signal = yield from e_signal
            m: Fold[Alg, D, E, Gradient[Pr]]
            m = pure(Gradient(signal.value @ infl.value))
            return m

        return next()


# lambda f: lambda pair: jax.jacrev(lambda a: f(a, pair[1]))(pair[0]),
class RTRL[Alg: _Infl[D, E, A, Pr, Lbl], D, E, A: Module, Pr: Module, Lbl, P](
    InfluenceTensorLearner[Alg, D, E, A, Pr, Lbl, P]
):

    type F[Dl, T] = Fold[Dl, D, E, T]
    type Grad_Pr = Gradient[Pr]
    type constraint = InfluenceTensorLearner.__infl__[E, A, Pr]

    @do()
    def influence_tensor[Dl: constraint](self, actvStep: F[Dl, IsVector[A]]):
        dl = yield from ProxyDl[Dl].askDl()
        influenceTensor = yield from dl.getInfluenceTensor()
        parametrized = self.reparameterize(actvStep)

        a0 = yield from dl.getActivation()
        p0 = yield from dl.getParameter()

        # I take the jvp instead of j @ v, bc I'd like to 1) do a hvp 2) forward over reverse is efficient: https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
        immedJacInflProduct: Array = jacobian_matrix_product(
            lambda a: toVector(parametrized(a, p0)[0]), a0, invmap(a0, lambda _: influenceTensor.value)
        )

        immedInfl: IsVector[Pr]
        env: E
        immedInfl, env = eqx.filter_jacfwd(lambda p: parametrized(a0, p), has_aux=True)(p0)

        _ = yield from put(env)
        m: Fold[Dl, D, E, Jacobian[Pr]]
        m = pure(Jacobian(immedJacInflProduct + toVector(immedInfl)))
        return m


class _RfL[D, E, A, Pr, Z](_Infl[D, E, A, Pr, Z], GetRfloConfig[E], Protocol): ...


class RFLO[Alg: _RfL[D, E, A, Pr, Z], D, E, A, Pr: Module, Z, P](InfluenceTensorLearner[Alg, D, E, A, Pr, Z, P]):
    type F[Dl, T] = Fold[Dl, D, E, T]
    type Grad_Pr = Gradient[Pr]

    class __rflo__(GetRfloConfig[E], InfluenceTensorLearner.__infl__[E, A, Pr], Protocol): ...

    @do()
    def influence_tensor[Dl: __rflo__](self, actvStep: F[Dl, IsVector[A]]):
        dl = yield from ProxyDl[Dl].askDl()
        alpha = yield from dl.getRfloConfig().fmap(lambda x: x.rflo_alpha)
        influenceTensor = yield from dl.getInfluenceTensor()
        a0 = yield from dl.getActivation()
        p0 = yield from dl.getParameter()
        parametrized = yield from self.reparameterize(actvStep)

        immedInfl: IsVector[Pr]
        env: E
        immedInfl, env = eqx.filter_jacrev(lambda p: parametrized(a0, p), has_aux=True)(p0)
        _ = yield from put(env)

        infl = (1 - alpha) * influenceTensor.value + alpha * toVector(immedInfl)
        m: Fold[Dl, D, E, Jacobian[Pr]]
        m = pure(Jacobian(infl))
        return m


class _UO[D, E, A, Pr, Z](
    _PfL[D, E, A, Pr, Z],
    GetUORO[E],
    PutUORO[E],
    GetRnnConfig[E],
    HasPRNG[E, jax.Array],
    Protocol,
): ...


class UORO[Alg: _UO[D, E, A, Pr, Z], D, E, A, Pr: Module, Z, P](PastFacingLearn[Alg, D, E, A, Pr, Z, P]):
    type F[Dl, T] = Fold[Dl, D, E, T]

    def __init__(self, distribution: Callable[[Array, tuple], Array]):
        super().__init__()
        self.distribution = distribution

    def gradientFlow(self, v: Array, actvStep: F[Alg, IsVector[A]]):
        @do()
        def next():
            dl = yield from ProxyDl[Alg].askDl()
            uoro = yield from dl.getUORO()

            a0 = yield from dl.getActivation()
            _ = yield from dl.putActivation(a0)
            p0 = yield from dl.getParameter()
            parametrized = yield from self.reparameterize(actvStep)
            A_prev = uoro.A  # I don't distinguish between column and row tensor.

            # 1 calculate the actv_jacobian vector product with A
            immedJac_A_product: Array
            immedJac_A_product = jvp(lambda a: toVector(parametrized(a, p0)[0]), a0, invmap(a0, lambda _: A_prev))

            # doing the vjp saves BIG on memory. Only use O(n^2) as we want
            fn: Callable[[IsVector[Pr]], tuple[IsVector[A], E]] = lambda p: parametrized(a0, p)
            _, vjp_func, env = eqx.filter_vjp(fn, p0, has_aux=True)
            _ = yield from put(env)

            (immedJacInfl_randomProj_,) = vjp_func(invmap(a0, lambda _: v))
            immedJacInfl_randomProj = toVector(immedJacInfl_randomProj_)

            immed_grads: Fold[Alg, D, E, tuple[Array, Array]]
            immed_grads = pure((immedJac_A_product, immedJacInfl_randomProj))
            return immed_grads

        return next()

    def creditAssign(self, A_: Array, B_: Array, e_signal: F[Alg, Gradient[A]]):
        def next(signal: Gradient[A]) -> Gradient[Pr]:
            q = signal.value @ A_
            return Gradient(q * B_)

        return e_signal.fmap(next)

    def recGrad(self, e_signal: F[Alg, Gradient[A]], actvStep: F[Alg, IsVector[A]]):
        @do()
        def next():
            dl = yield from ProxyDl[Alg].askDl()
            uoro = yield from dl.getUORO()
            rnnConfig = yield from dl.getRnnConfig()
            key = yield from dl.updatePRNG()
            v = self.distribution(key, (rnnConfig.n_h,))
            B_prev = uoro.B

            immedJac_A_product, immedJacInfl_randomProj = yield from self.gradientFlow(v, actvStep)

            rho0: Array = jnp.sqrt(jnp.linalg.norm(B_prev) / jnp.linalg.norm(immedJac_A_product))
            rho1: Array = jnp.sqrt(jnp.linalg.norm(immedJacInfl_randomProj) / jnp.linalg.norm(v))

            A_new = rho0 * immedJac_A_product + rho1 * v
            B_new = rho0 * B_prev + rho1 * immedJacInfl_randomProj
            _ = yield from dl.putUORO(replace(uoro, A=A_new, B=B_new))

            return self.creditAssign(A_new, B_new, e_signal)

        return next()


@eqx.filter_jit
def trainStep[
    Dl, D, E
](learner: Fold[Dl, D, E, Unit], dialect: Dl, t_series: Traversable[D], initEnv: E,) -> E:
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
