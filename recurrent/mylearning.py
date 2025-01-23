from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any, Callable, Generic, TypeVar, Protocol, Iterable, cast
from donotation import do
import torch
from torch import Tensor
from torch.utils import _pytree as pytree
from torch.utils._pytree import PyTree
import torch.func as Tfn

from recurrent.objectalgebra.typeclasses import (
    GetActivation,
    GetHyperParameter,
    GetInfluenceTensor,
    GetParameter,
    GetRfloConfig,
    GetRnnConfig,
    HasInput,
    HasPredictionInput,
    HasLabel,
    PutActivation,
    PutInfluenceTensor,
    PutParameter,
    GetUORO,
    PutUORO,
)

from recurrent.mytypes import *
from recurrent.monad import *
from recurrent.parameters import RnnParameter, SgdParameter


type LossFn[A, B] = Callable[[A, B], LOSS]


def pytree_norm(tree):
    leaves, _ = pytree.tree_flatten(tree)
    return torch.sqrt(sum(torch.sum(leaf**2) for leaf in leaves))


def jvp(f, primal, tangent):
    return torch.func.jvp(f, (primal,), (tangent,))[1]


def jacobian_matrix_product(f, primal, matrix):
    # jvp = lambda p, t: torch.func.jvp(f, (p,), (t,))[1]
    wrapper = lambda p, t: jvp(f, p, t)
    return torch.vmap(wrapper, in_dims=(None, 1), out_dims=1)(primal, matrix)


class _SGD[E, Pr](
    GetParameter[E, Pr],
    PutParameter[E, Pr],
    GetHyperParameter[E, SgdParameter],
    Protocol,
): ...


@do()
def doSgdStep[D, E, Pr: PYTREE](grad: Gradient[Pr]) -> G[Fold[_SGD[E, Pr], D, E, Unit]]:
    dl = yield from ProxyDl[_SGD[E, Pr]].askDl()
    param = yield from dl.getParameter()
    hyperparam = yield from dl.getHyperParameter()
    new_param = pytree.tree_map(
        lambda x, y: x - hyperparam.learning_rate * y, param, grad.value
    )
    return dl.putParameter(new_param)


class _R[D, E](
    GetActivation[E, ACTIVATION],
    PutActivation[E, ACTIVATION],
    GetParameter[E, RnnParameter],
    HasInput[D, torch.Tensor],
    GetRnnConfig[E],
    Protocol,
): ...


@do()
def doRnnStep[D, E]() -> G[Fold[_R[D, E], D, E, ACTIVATION]]:
    dl = yield from ProxyDl[_R[D, E]].askDl()
    x = yield from dl.getInput()
    a = yield from dl.getActivation()
    param = yield from dl.getParameter()
    cfg = yield from dl.getRnnConfig()
    w_rec = torch.reshape(param.w_rec, (cfg.n_h, cfg.n_h + cfg.n_in + 1))
    a_rec = w_rec @ torch.cat((a, x, torch.tensor([1.0])))
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
def doRnnReadout[D, E]() -> G[Fold[_Re[E], D, E, PREDICTION]]:
    dl = yield from ProxyDl[_Re[E]].askDl()
    a = yield from dl.getActivation()
    param = yield from dl.getParameter()
    cfg = yield from dl.getRnnConfig()
    w_out = torch.reshape(param.w_out, (cfg.n_out, cfg.n_h + 1))
    return pure(PREDICTION(w_out @ torch.cat((a, torch.tensor([1.0])))))


class _L[D, T](HasLabel[D, T], Protocol): ...


type DoLossFn[D, E, P, T] = Callable[[P], Fold[_L[D, T], D, E, LOSS]]


def doLoss[D, E, P, T](lossFn: LossFn[T, P]) -> DoLossFn[D, E, P, T]:
    @do()
    def loss_(pred: P) -> G[Fold[_L[D, T], D, E, LOSS]]:
        d = yield from ProxyDl[_L[D, T]].askDl()
        label = yield from d.getLabel()
        loss = lossFn(pred, label)
        return pure(loss)

    return loss_


def doGradient[
    Dl, D, E, Pr, T
](
    lossM: Fold[Dl, D, E, T],
    read_wrt: Fold[Dl, D, E, Pr],
    write_wrt: Callable[[Pr], Fold[Dl, D, E, Unit]],
):
    @do()
    def doGradient_() -> G[Fold[Dl, D, E, Gradient[Pr]]]:
        dl = yield from ProxyDl[Dl].askDl()
        d = yield from ProxyR[D].ask()
        e = yield from ProxyS[E].get()
        pr = yield from read_wrt

        def parametrized(p: Pr) -> tuple[T, E]:
            return write_wrt(p).then(lossM).func(dl, d, e)

        gr: Pr
        env: E
        gr, env = Tfn.jacrev(parametrized, has_aux=True)(pr)
        _ = yield from put(env)
        return pure(Gradient(gr))

    return doGradient_()


@do()
def doAvgGradient[
    DL, D, E, Pr: PYTREE
](
    step: Fold[DL, D, E, Gradient[Pr]],
    weightFn: Callable[[D], float],
    gr0: Gradient[Pr],
) -> G[Fold[DL, Iterable[D], E, Gradient[Pr]]]:

    @do()
    def doWeight(gr_accum: Gradient[Pr]) -> G[Fold[DL, D, E, Gradient[Pr]]]:
        d = yield from ProxyR[D].ask()
        gr_new = yield from step
        gr_accum_new: Pr
        gr_accum_new = pytree.tree_map(
            lambda x, y: x + y * weightFn(d),
            gr_accum.value,
            gr_new.value,
        )
        return pure(Gradient(gr_accum_new))

    return foldM(doWeight, gr0)


@do()
def doBatchGradients[
    Dl, D: PYTREE | Tensor, E, Pr: PYTREE
](step: Fold[Dl, D, E, Gradient[Pr]], e_dim: E) -> G[Fold[Dl, D, E, Gradient[Pr]]]:
    dl = yield from ProxyDl[Dl].askDl()
    run: Callable[[D, E], tuple[Gradient[Pr], E]]
    run = lambda d, e: step.func(dl, d, e)
    gr_ = yield from toFold(torch.vmap(run, in_dims=(0, e_dim), out_dims=(0, e_dim)))
    gr = cast(Gradient[Pr], gr_)
    gr_summed: Pr
    gr_summed = pytree.tree_map(lambda x: torch.sum(x, dim=0), gr.value)
    return pure(Gradient(gr_summed))


class RnnLibrary[DL, D, E, P, Pr](NamedTuple):
    rnn: Fold[DL, D, E, P]
    rnnWithLoss: Fold[DL, D, E, LOSS]
    rnnWithGradient: Fold[DL, D, E, Gradient[Pr]]

    type UpdateP = Callable[[Gradient[Pr]], Fold[DL, D, E, Unit]]

    def trainStep(self, updateParam: UpdateP) -> Fold[DL, Iterable[D], E, Unit]:
        return repeatM(self.rnnWithGradient.flat_map(updateParam))


class _OfL[D, E, A, Pr, X, Y, Z](
    GetActivation[E, A],
    PutActivation[E, A],
    PutParameter[E, Pr],
    GetParameter[E, Pr],
    HasInput[D, X],
    HasPredictionInput[D, Y],
    HasLabel[D, Z],
    Protocol,
): ...


# GradientLibrary[ENV, Iterable[PRED], Iterable[DATA]]:
def offlineLearning[
    D, E, A, Pr, X, Y, Z, P
](
    activationStep: Fold[_OfL[D, E, A, Pr, X, Y, Z], D, E, A],
    predictionStep: Fold[_OfL[D, E, A, Pr, X, Y, Z], D, E, P],
    computeLoss: LossFn[Z, P],
) -> RnnLibrary[_OfL[D, E, A, Pr, X, Y, Z], Iterable[D], E, Iterable[P], Pr]:
    type DL = _OfL[D, E, A, Pr, X, Y, Z]

    rnnStep = activationStep.then(predictionStep)

    rnn = traverse(rnnStep)

    @do()
    def rnnWithLoss():
        @do()
        def accumFn(accum: LOSS) -> G[Fold[DL, D, E, LOSS]]:
            loss = yield from rnnStep.flat_map(doLoss(computeLoss))
            return pure(LOSS(accum + loss))

        return foldM(accumFn, LOSS(torch.tensor(0)))

    @do()
    def rnnWithGradient():
        dl = yield from ProxyDl[DL].askDl()
        return doGradient(rnnWithLoss(), dl.getParameter(), dl.putParameter)

    return RnnLibrary[DL, Iterable[D], E, Iterable[P], Pr](
        rnn=rnn,
        rnnWithLoss=rnnWithLoss(),
        rnnWithGradient=rnnWithGradient(),
    )


class _PfL[D, E, A, Pr, Z](
    GetActivation[E, A],
    PutActivation[E, A],
    PutParameter[E, Pr],
    GetParameter[E, Pr],
    HasLabel[D, Z],
    Protocol,
): ...


class PastFacingLearn[Alg: _PfL[D, E, A, Pr, Z], D, E, A, Pr: PYTREE, Z, P](ABC):

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

        immedL = predictionStep.flat_map(doLoss(computeLoss))

        @do()
        def total_grad():
            dl = yield from ProxyDl[Alg].askDl()

            # state issue, updating order is all wrong
            e_signal = doGradient(immedL, dl.getActivation(), dl.putActivation)

            grad_rec = yield from self.creditAndUpdateState(e_signal, activationStep)
            # order matters, need to update actv b4 grad flow the readout
            grad_o = yield from doGradient(immedL, dl.getParameter(), dl.putParameter)

            gradient: Gradient[Pr]
            gradient = pytree.tree_map(lambda x, y: x + y, grad_o, grad_rec)
            m: Fold[Alg, D, E, Gradient[Pr]]
            m = pure(gradient)

            return m

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


class InfluenceTensorLearner[Alg: _Infl[D, E, A, Pr, Z], D, E, A, Pr: PYTREE, Z, P](
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
            ca = pytree.tree_map(lambda x: signal.value @ x, infl)
            m: Fold[Alg, D, E, Gradient[Pr]]
            m = pure(ca)

            return m

        return next()


class RTRL[Alg: _Infl[D, E, A, Pr, Z], D, E, A, Pr: PYTREE, Z, P](
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
        immedJacInflProduct: Gradient[Pr]
        immedJacInflProduct = pytree.tree_map(
            lambda x: jacobian_matrix_product(lambda a: parametrized(a, p0)[0], a0, x),
            influenceTensor,
        )

        # performance below really depends on the problem. For RTRL, I should use jacrev. For OHO, jacfwd. Will make this configurable later.
        immedInfl: Pr
        env: E
        immedInfl, env = Tfn.jacfwd(parametrized, argnums=1, has_aux=True)(a0, p0)

        infl: Pr
        infl = pytree.tree_map(lambda x, y: x + y, immedInfl, immedJacInflProduct.value)
        _ = yield from put(env)
        m: Fold[Dl, D, E, Gradient[Pr]]
        m = pure(Gradient(infl))
        return m


class _RfL[D, E, A, Pr, Z](_Infl[D, E, A, Pr, Z], GetRfloConfig[E], Protocol): ...


class RFLO[Alg: _RfL[D, E, A, Pr, Z], D, E, A, Pr: PYTREE, Z, P](
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
        immedInfl, env = Tfn.jacrev(parametrized, argnums=1, has_aux=True)(a0, p0)

        infl: Pr
        infl = pytree.tree_map(
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
    Protocol,
): ...


class UORO[Alg: _UO[D, E, A, Pr, Z], D, E, A, Pr: PYTREE, Z, P](
    PastFacingLearn[Alg, D, E, A, Pr, Z, P]
):
    type F[Dl, T] = Fold[Dl, D, E, T]

    def gradientFlow(self, v: Tensor, actvStep: F[Alg, A]):
        @do()
        def next():
            dl = yield from ProxyDl[Alg].askDl()
            uoro = yield from dl.getUORO()

            a0 = yield from dl.getActivation()
            p0 = yield from dl.getParameter()
            parametrized = yield from self.reparameterize(actvStep)
            A_prev = uoro.A  # I don't distinguish between column and row tensor.

            # 1 calculate the actv_jacobian vector product with A
            immedJac_A_product: Tensor
            immedJac_A_product = jvp(lambda a: parametrized(a, p0)[0], a0, A_prev)

            # doing the vjp saves BIG on memory. Only use O(n^2) as we want
            env: E
            _, vjp_func, env = Tfn.vjp(lambda p: parametrized(a0, p), p0, has_aux=True)
            _ = yield from put(env)

            immedJacInfl_randomProj: Pr
            (immedJacInfl_randomProj,) = vjp_func(v)

            immed_grads: Fold[Alg, D, E, tuple[Tensor, Pr]]
            immed_grads = pure((immedJac_A_product, immedJacInfl_randomProj))
            return immed_grads

        return next()

    def creditAndUpdateState(self, e_signal: F[Alg, Gradient[A]], actvStep: F[Alg, A]):
        @do()
        def next():
            dl = yield from ProxyDl[Alg].askDl()
            uoro = yield from dl.getUORO()
            rnnConfig = yield from dl.getRnnConfig()
            #!!! WARNING, IMPURITY IN PROGRESS, will address later. Just don't run this function more than once, if autodiffing through it
            v = torch.distributions.uniform.Uniform(-1, 1).sample((rnnConfig.n_h,))
            B_prev = uoro.B

            immedJac_A_product, immedJacInfl_randomProj = yield from self.gradientFlow(
                v, actvStep
            )

            rho0: Tensor = torch.sqrt(
                pytree_norm(B_prev) / torch.norm(immedJac_A_product)
            )
            rho1: Tensor = torch.sqrt(
                pytree_norm(immedJacInfl_randomProj) / torch.norm(v)
            )

            A_new = rho0 * immedJac_A_product + rho1 * v
            B_new: Gradient[Pr] = Gradient(
                pytree.tree_map(
                    lambda x, y: x / rho0 + y / rho1,
                    B_prev.value,
                    immedJacInfl_randomProj,
                )
            )

            _ = yield from dl.putUORO(replace(uoro, A=A_new, B=B_new))

            signal = yield from e_signal  # order matters, should be after actv
            q = signal.value @ A_new
            ca: Fold[Alg, D, E, Gradient[Pr]]
            ca = pure(pytree.tree_map(lambda x: x * q, B_new))
            return ca

        return next()
