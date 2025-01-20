from abc import ABC, abstractmethod
from dataclasses import dataclass
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
    HasInput,
    HasPredictionInput,
    HasLabel,
    PutActivation,
    PutInfluenceTensor,
    PutParameter,
)

from recurrent.mytypes import *
from recurrent.monad import *
from recurrent.parameters import RnnParameter


# STATEM = Callable[[DATA, ENV], ENV]
# ENDOMORPHIC = Callable[[T, T], T]
# LOSSFN = Callable[[DATA, ENV], LOSS]

# ACTIV = TypeVar("ACTIV")
# PRED = TypeVar("PRED")
# PARAM = TypeVar("PARAM")
# HPARAM = TypeVar("HPARAM")
# PARAM_T = TypeVar("PARAM_T", covariant=True)
# PARAM_R = TypeVar("PARAM_R", covariant=True)
# PARAM_O = TypeVar("PARAM_O", covariant=True)

# DATA_L = TypeVar("DATA_L", contravariant=True)
# X_L = TypeVar("X_L", covariant=True)
# Y_L = TypeVar("Y_L", covariant=True)
# Z_L = TypeVar("Z_L", covariant=True)

# ACTIV_CO = TypeVar("ACTIV_CO", covariant=True)
# PRED_CO = TypeVar("PRED_CO", covariant=True)
# PRED_CON = TypeVar("PRED_CON", contravariant=True)
# HPARAM_CO = TypeVar("HPARAM_CO", covariant=True)
# ENV_CON = TypeVar("ENV_CON", contravariant=True)
# PARAM_CON = TypeVar("PARAM_CON", contravariant=True)
# PARAM_CO = TypeVar("PARAM_CO", covariant=True)

# PARAM_TENSOR = TypeVar("PARAM_TENSOR", bound=torch.Tensor | PYTREE)
# ACTIV_TENSOR = TypeVar("ACTIV_TENSOR", bound=torch.Tensor | PYTREE)
# T_TENSOR = TypeVar("T_TENSOR", bound=torch.Tensor | PYTREE)

# DATA_NEW = TypeVar("DATA_NEW")

# T_CON = TypeVar("T_CON", contravariant=True)
# T_CO = TypeVar("T_CO", covariant=True)


# # literally State get
# def reparameterizeOutput(
#     step: Callable[[DATA, ENV], ENV], read: Callable[[ENV], X]
# ) -> Callable[[DATA, ENV], X]:
#     def reparametrized(info: DATA, env: ENV) -> X:
#         env_ = step(info, env)
#         return read(env_)

#     return reparametrized


# # literally State put
# def reparametrizeInput(
#     step: Callable[[DATA, ENV], Z], writer: Callable[[X, ENV], ENV]
# ) -> Callable[[DATA, ENV, X], Z]:
#     def reparametrized(info: DATA, env: ENV, x: X) -> Z:
#         env_ = writer(x, env)
#         return step(info, env_)

#     return reparametrized


# class _Activation(
#     GetActivation[ENV, ACTIV],
#     PutActivation[ENV, ACTIV],
#     GetParameter[ENV, PARAM_CO],
#     Protocol[ENV, ACTIV, PARAM_CO],
# ):
#     pass


# class _ActivationData(HasInput[DATA_L, X_L], Protocol[DATA_L, X_L]):
#     pass


# def activation(
#     dialect: _Activation[ENV, ACTIV, PARAM],
#     dataDialect: _ActivationData[DATA, X],
#     step: Callable[[X, ACTIV, PARAM], ACTIV],
# ) -> ReaderState[DATA, ENV, Unit]:
#     def update(pair: tuple[DATA, ENV]) -> ENV:
#         data, env = pair
#         a = dialect.getActivation(env)
#         x = dataDialect.getInput(data)
#         p = dialect.getParameter(env)
#         return dialect.putActivation(step(x, a, p), env)

#     return (
#         ReaderState[DATA, ENV, Unit]
#         .ask_get()
#         .fmap(update)
#         .bind(ReaderState[DATA, ENV, Unit].put)
#     )


# class _Prediction(
#     GetActivation[ENV_CON, ACTIV_CO],
#     GetParameter[ENV_CON, PARAM_CO],
#     Protocol[ENV_CON, ACTIV_CO, PARAM_CO],
# ):
#     pass


# class _PredictionData(HasPredictionInput[DATA_L, X_L], Protocol[DATA_L, X_L]):
#     pass


# def prediction(
#     dialect: _Prediction[ENV, ACTIV, PARAM],
#     dataDialect: _PredictionData[DATA, X],
#     step: Callable[[X, ACTIV, PARAM], PRED],
# ) -> ReaderState[DATA, ENV, PRED]:
#     def update(pair: tuple[DATA, ENV]) -> PRED:
#         data, env = pair
#         x = dataDialect.getPredictionInput(data)
#         a = dialect.getActivation(env)
#         return step(x, a, dialect.getParameter(env))

#     return ReaderState[DATA, ENV, tuple[DATA, ENV]].ask_get().fmap(update)

# x: torch.Tensor, a: ACTIVATION, param: RnnParameter

type LossFn[A, B] = Callable[[A, B], LOSS]


def jacobian_matrix_product(f, primal, matrix):
    return torch.vmap(torch.func.jvp, in_dims=(None, None, 0))(f, primal, matrix)[1]


class _R[D, E](
    GetActivation[E, ACTIVATION],
    PutActivation[E, ACTIVATION],
    GetParameter[E, RnnParameter],
    HasInput[D, torch.Tensor],
    Protocol,
): ...


@do()
def doRnnStep[D, E]() -> G[Fold[_R[D, E], D, E, ACTIVATION]]:
    dl = yield from askDl(Proxy[_R[D, E]]())
    x = yield from dl.getInput()
    a = yield from dl.getActivation()
    param = yield from dl.getParameter()
    a_rec = param.w_rec @ torch.cat((x, a, torch.tensor([1.0])))
    a_new = ACTIVATION((1 - param.alpha) * a + param.alpha * param.activationFn(a_rec))
    _ = yield from dl.putActivation(a_new)
    return pure(a_new)


class _Re[E](
    GetActivation[E, ACTIVATION],
    GetParameter[E, RnnParameter],
    Protocol,
): ...


@do
def doRnnReadout[D, E]() -> G[Fold[_Re[E], D, E, PREDICTION]]:
    dl = yield from askDl(Proxy[_Re[E]]())
    a = yield from dl.getActivation()
    param = yield from dl.getParameter()
    return pure(PREDICTION(param.w_out @ torch.cat((a, torch.tensor([1.0])))))


class _L[D, T](HasLabel[D, T], Protocol): ...


type DoLossFn[D, E, P, T] = Callable[[P], Fold[_L[D, T], D, E, LOSS]]


def doLoss[D, E, P, T](lossFn: LossFn[T, P]) -> DoLossFn[D, E, P, T]:
    @do()
    def loss_(pred: P) -> G[Fold[_L[D, T], D, E, LOSS]]:
        d = yield from askDl(Proxy[_L[D, T]]())
        label = yield from d.getLabel()
        return pure(lossFn(label, pred))

    return loss_


def doGradient[
    Dl, D, E, Pr: Tensor
](
    lossM: Fold[Dl, D, E, LOSS],
    read_wrt: Fold[Dl, D, E, Pr],
    write_wrt: Callable[[Pr], Fold[Dl, D, E, Unit]],
):
    @do()
    def doGradient() -> G[Fold[Dl, D, E, Gradient[Pr]]]:
        dl = yield from askDl(Proxy[Dl]())
        d = yield from ask(Proxy[D]())
        e = yield from get(Proxy[E]())
        pr = yield from read_wrt

        def parametrized(p: Pr) -> tuple[LOSS, E]:
            return write_wrt(p).then(lossM).func(dl, d, e)

        gr: Pr
        env: E
        gr, env = Tfn.jacrev(parametrized, has_aux=True)(pr)
        _ = yield from put(env)
        return pure(Gradient[Pr](gr))

    return doGradient()


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
        d = yield from ask(Proxy[D]())
        gr_new = yield from step
        gr_accum_new: Pr
        gr_accum_new = pytree.tree_map(
            lambda x, y: x + y * weightFn(d),
            gr_accum.value,
            gr_new.value,
        )
        return pure(Gradient[Pr](gr_accum_new))

    return foldM(doWeight, gr0)


@do()
def doBatchGradients[
    Dl, D: PYTREE | Tensor, E, Pr: PYTREE
](step: Fold[Dl, D, E, Gradient[Pr]], e_dim: E) -> G[Fold[Dl, D, E, Gradient[Pr]]]:
    dl = yield from askDl(Proxy[Dl]())
    run: Callable[[D, E], tuple[Gradient[Pr], E]]
    run = lambda d, e: step.func(dl, d, e)
    gr_ = yield from toFold(torch.vmap(run, in_dims=(0, e_dim), out_dims=(0, e_dim)))
    gr = cast(Gradient[Pr], gr_)
    gr_summed: Pr
    gr_summed = pytree.tree_map(lambda x: torch.sum(x, dim=0), gr.value)
    return pure(Gradient[Pr](gr_summed))


class RnnLibrary[DL, D, E, P, Pr](NamedTuple):
    rnn: Fold[DL, D, E, P]
    rnnWithLoss: Fold[DL, D, E, LOSS]
    rnnWithGradient: Fold[DL, D, E, Gradient[Pr]]


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
) -> RnnLibrary[_OfL[D, E, A, Pr, X, Y], Iterable[D], E, Iterable[P], Pr]:
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
        dl = yield from askDl(Proxy[DL]())
        return doGradient(rnnWithLoss(), dl.getParameter(), dl.putParameter)

    return RnnLibrary[DL, Iterable[D], E, Iterable[P], Pr](
        rnn=rnn,
        rnnWithLoss=rnnWithLoss(),
        rnnWithGradient=rnnWithGradient(),
    )


class _PfL[D, E, A, Pr, X, Y, Z](
    GetActivation[E, A],
    PutActivation[E, A],
    PutParameter[E, Pr],
    GetParameter[E, Pr],
    GetInfluenceTensor[E, Gradient[Pr]],
    PutInfluenceTensor[E, Gradient[Pr]],
    HasInput[D, X],
    HasPredictionInput[D, Y],
    HasLabel[D, Z],
    Protocol,
): ...


class PastFacingLearn[D, E, A: Tensor, Pr: PYTREE, X, Y, Z, P](ABC):
    type DL = _PfL[D, E, A, Pr, X, Y, Z]

    @abstractmethod
    def influence_tensor(
        self, activationStep: Fold[DL, D, E, A]
    ) -> Fold[DL, D, E, Gradient[Pr]]:
        pass

    @staticmethod
    @do()
    def reparameterize(
        activationStep: Fold[DL, D, E, A]
    ) -> G[Fold[DL, D, E, Callable[[A, Pr], tuple[A, E]]]]:
        type DL = _PfL[D, E, A, Pr, X, Y, Z]
        dl = yield from askDl(Proxy[DL]())
        d = yield from ask(Proxy[D]())
        e = yield from get(Proxy[E]())

        def parametrized(a: A, pr: Pr) -> tuple[A, E]:
            return (
                dl.putActivation(a)
                .then(dl.putParameter(pr))
                .then(activationStep)
                .func(dl, d, e)
            )

        return pure(parametrized)

    def onlineLearning(
        self,
        activationStep: Fold[DL, D, E, A],
        predictionStep: Fold[DL, D, E, P],
        computeLoss: LossFn[Z, P],
    ) -> RnnLibrary[DL, D, E, P, Pr]:
        type DL = _PfL[D, E, A, Pr, X, Y, Z]

        immedL = predictionStep.flat_map(doLoss(computeLoss))

        @do()
        def creditAssignment(
            infl: Gradient[Pr],
        ) -> G[Fold[DL, D, E, Gradient[Pr]]]:
            dl = yield from askDl(Proxy[DL]())
            _ = yield from dl.putInfluenceTensor(infl)
            immed: Gradient[A]
            immed = yield from doGradient(immedL, dl.getActivation(), dl.putActivation)
            ca: Gradient[Pr]
            ca = pytree.tree_map(lambda x: immed.value @ x, infl)
            return pure(ca)

        gradient = self.influence_tensor(activationStep).flat_map(creditAssignment)
        RnnLibrary[DL, D, E, P, Pr](
            rnn=activationStep.then(predictionStep),
            rnnWithLoss=activationStep.then(immedL),
            rnnWithGradient=gradient,
        )


class RTRL[D, E, A: Tensor, Pr: PYTREE, X, Y, Z, P](
    PastFacingLearn[D, E, A, Pr, X, Y, Z, P]
):
    type DL = _PfL[D, E, A, Pr, X, Y, Z]

    @do()
    def influence_tensor(
        self, activationStep: Fold[DL, D, E, A]
    ) -> G[Fold[DL, D, E, Gradient[Pr]]]:
        type DL = _PfL[D, E, A, Pr, X, Y, Z]
        dl = yield from askDl(Proxy[DL]())
        influenceTensor = yield from dl.getInfluenceTensor()
        a0 = yield from dl.getActivation()
        p0 = yield from dl.getParameter()
        parametrized = yield from self.reparameterize(activationStep)

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
        return pure(Gradient[Pr](infl))


class _RfL[D, E, A, Pr, X, Y, Z](
    _PfL[D, E, A, Pr, X, Y, Z], GetRfloConfig[E], Protocol
): ...


class RFLO[D, E, A: Tensor, Pr: PYTREE, X, Y, Z, P](
    PastFacingLearn[D, E, A, Pr, X, Y, Z, P]
):
    type DL = _RfL[D, E, A, Pr, X, Y, Z]

    @do()
    def influence_tensor(
        self, activationStep: Fold[DL, D, E, A]
    ) -> G[Fold[DL, D, E, Gradient[Pr]]]:
        type DL = _RfL[D, E, A, Pr, X, Y, Z]
        dl = yield from askDl(Proxy[DL]())
        alpha = yield from dl.getRfloConfig().fmap(lambda x: x.rflo_alpha)
        influenceTensor = yield from dl.getInfluenceTensor()
        a0 = yield from dl.getActivation()
        p0 = yield from dl.getParameter()
        parametrized = yield from self.reparameterize(activationStep)

        immedInfl: Pr
        env: E
        immedInfl, env = Tfn.jacrev(parametrized, argnums=1, has_aux=True)(a0, p0)
        infl: Pr
        infl = pytree.tree_map(
            lambda x, y: (1 - alpha) * x + alpha * y, influenceTensor.value, immedInfl
        )
        _ = yield from put(env)
        return pure(Gradient[Pr](infl))


# def SGD(learning_rate: float) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
#     def SGD_(param: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
#         return param - learning_rate * grad
#     return SGD_


# def getActivationFn(case: ActivationFnType) -> Callable[[torch.Tensor], torch.Tensor]:
#     match case:
#         case RELU():
#             return torch.relu
#         case TANH():
#             return torch.tanh
#         case _:
#             raise ValueError('Activation function not supported')


# def activationLayersTrans(activationFn: Callable[[torch.Tensor], torch.Tensor]):
#     def activationTrans_(t: Union[HasActivation[MODEL, List[torch.Tensor]], HasParameter[MODEL, PARAM]]) -> Callable[[torch.Tensor, MODEL], MODEL]:
#         def activationTrans__(x: torch.Tensor, env: MODEL) -> MODEL:
#             as_ = t.getActivation(env)
#             W_rec, W_in, b_rec, _, _, alpha = t.getParameter(env)
#             def scanner(prevActv: torch.Tensor, nextActv: torch.Tensor) -> torch.Tensor:  # i'm folding over nextActv
#                 return rnnTrans(activationFn)(prevActv, (W_in, W_rec, b_rec, alpha), nextActv)
#             as__ = list(scan0(scanner, x, as_))
#             return t.putActivation(as__, env)
#         return activationTrans__
#     return activationTrans_
# doing multiple layers is just a fold over it


# def minitest(
#     dialect: _RnnDialect[ENV, ACTIV, PRED, PARAM, PARAM_R, PARAM_O], env: ENV
# ) -> ACTIV:
#     return RnnLearnableInterpreter[ENV].getActivation(env)


# temp: RnnBPTTState = RnnBPTTState(
#     activation=ACTIVATION(torch.tensor([1.0])),
#     parameter=PARAMETER(torch.tensor([1.0])),
#     trainPrediction=PREDICTION(torch.tensor([1.0])),
#     trainLoss=LOSS(torch.tensor([1.0])),
#     trainGradient=GRADIENT(torch.tensor([1.0])),
#     hyperparameter=HYPERPARAMETER(torch.tensor([1.0])),
#     n_h=1,
#     n_in=1,
#     n_out=1,
#     alpha=1.0,
#     activationFn=lambda x: x,
# )

# test = minitest(RnnLearnableInterpreter[RnnBPTTState], temp)
