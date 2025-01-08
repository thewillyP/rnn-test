from dataclasses import dataclass
from typing import Callable, Generic, TypeVar, Protocol, Iterable
import torch
from myobjectalgebra import (
    GetActivation,
    GetGradient,
    GetHyperParameter,
    GetInfluenceTensor,
    GetLoss,
    GetParameter,
    GetPrediction,
    HasReadoutWeights,
    HasInput,
    HasPredictionInput,
    HasLabel,
    HasRecurrentWeights,
    PutActivation,
    PutGradient,
    PutInfluenceTensor,
    PutLoss,
    PutParameter,
    PutPrediction,
)
from myfunc import collapseF, flip, foldr, scan0, fst, snd, liftA2Compose, foldl
import itertools
from mytypes import *
from operator import add
from collections.abc import Collection


from myrecords import RnnBPTTState

STATEM = Callable[[DATA, ENV], ENV]
ENDOMORPHIC = Callable[[T, T], T]
LOSSFN = Callable[[DATA, ENV], LOSS]

ACTIV = TypeVar("ACTIV")
PRED = TypeVar("PRED")
PARAM = TypeVar("PARAM")
HPARAM = TypeVar("HPARAM")
PARAM_T = TypeVar("PARAM_T", covariant=True)
PARAM_R = TypeVar("PARAM_R", covariant=True)
PARAM_O = TypeVar("PARAM_O", covariant=True)

DATA_L = TypeVar("DATA_L", contravariant=True)
X_L = TypeVar("X_L", covariant=True)
Y_L = TypeVar("Y_L", covariant=True)
Z_L = TypeVar("Z_L", covariant=True)

ACTIV_CO = TypeVar("ACTIV_CO", covariant=True)
PRED_CO = TypeVar("PRED_CO", covariant=True)
PRED_CON = TypeVar("PRED_CON", contravariant=True)
HPARAM_CO = TypeVar("HPARAM_CO", covariant=True)

PARAM_TENSOR = TypeVar("PARAM_TENSOR", bound=torch.Tensor)
ACTIV_TENSOR = TypeVar("ACTIV_TENSOR", bound=torch.Tensor)

DATA_NEW = TypeVar("DATA_NEW")


def jacobian_matrix_product(f, primal, matrix):
    return torch.vmap(torch.func.jvp, in_dims=(None, None, 0))(f, primal, matrix)[1]


# literally State get
def reparameterizeOutput(
    step: Callable[[DATA, ENV], ENV], read: Callable[[ENV], X]
) -> Callable[[DATA, ENV], X]:
    def reparametrized(info: DATA, env: ENV) -> X:
        env_ = step(info, env)
        return read(env_)

    return reparametrized


# literally State put
def reparametrizeInput(
    step: Callable[[DATA, ENV], Z], writer: Callable[[X, ENV], ENV]
) -> Callable[[DATA, ENV, X], Z]:
    def reparametrized(info: DATA, env: ENV, x: X) -> Z:
        env_ = writer(x, env)
        return step(info, env_)

    return reparametrized


def fmapState(
    reader: Callable[[ENV], X],
    writer: Callable[[X, ENV], ENV],
    f: Callable[[X], X],
) -> Callable[[DATA, ENV], ENV]:
    def fmapped(_: DATA, env: ENV) -> ENV:
        x = reader(env)
        x_ = f(x)
        env_ = writer(x_, env)
        return env_

    return fmapped


def makeLossPair(dialect: GetLoss[ENV, LOSS]) -> Callable[[ENV], tuple[LOSS, ENV]]:
    def lossPair(env: ENV) -> tuple[LOSS, ENV]:
        return dialect.getLoss(env), env

    return lossPair


class _Activation(
    GetActivation[ENV, ACTIV],
    PutActivation[ENV, ACTIV],
    GetParameter[ENV, PARAM],
    HasRecurrentWeights[ENV, PARAM, PARAM_T],
    Protocol[ENV, ACTIV, PARAM, PARAM_T],
):
    pass


class _ActivationData(HasInput[DATA_L, X_L], Protocol[DATA_L, X_L]):
    pass


def activation(
    dialect: _Activation[ENV, ACTIV, PARAM, PARAM_T],
    dataDialect: _ActivationData[DATA, X],
    step: Callable[[X, ACTIV, PARAM_T], ACTIV],
    info: DATA,
    env: ENV,
) -> ENV:
    a = dialect.getActivation(env)
    x = dataDialect.getInput(info)
    w_rec = dialect.getRecurrentWeights(dialect.getParameter(env), env)
    a_ = step(x, a, w_rec)
    return dialect.putActivation(a_, env)


class _Prediction(
    GetActivation[ENV, ACTIV_CO],
    GetParameter[ENV, PARAM],
    HasReadoutWeights[ENV, PARAM, PARAM_T],
    PutPrediction[ENV, PRED_CON],
    Protocol[ENV, ACTIV_CO, PARAM, PARAM_T, PRED_CON],
):
    pass


class _PredictionData(HasPredictionInput[DATA_L, X_L], Protocol[DATA_L, X_L]):
    pass


# oho readout doesnt use hyperparameter as its readout though...
def prediction(
    dialect: _Prediction[ENV, ACTIV, PARAM, PARAM_T, PRED],
    dataDialect: _PredictionData[DATA, X],
    step: Callable[[X, ACTIV, PARAM_T], PRED],
    info: DATA,
    env: ENV,
) -> ENV:
    a = dialect.getActivation(env)
    x = dataDialect.getPredictionInput(info)
    w_out = dialect.getReadoutWeights(dialect.getParameter(env), env)
    pred = step(x, a, w_out)
    return dialect.putPrediction(pred, env)


class _Loss(
    GetLoss[ENV, LOSS],
    PutLoss[ENV, LOSS],
    GetPrediction[ENV, PRED_CO],
    Protocol[ENV, PRED_CO],
):
    pass


def loss(
    dialect: _Loss[ENV, PRED],
    dataDialect: HasLabel[DATA, Y],
    computeLoss: Callable[[Y, PRED], LOSS],
    foldLoss: Callable[[LOSS, LOSS], LOSS],
    info: DATA,
    env: ENV,
) -> ENV:
    y = dataDialect.getLabel(info)
    pred = dialect.getPrediction(env)
    loss_ = foldLoss(computeLoss(y, pred), dialect.getLoss(env))
    return dialect.putLoss(loss_, env)


def computeGradient(
    lossFn: Callable[[DATA, ENV], tuple[LOSS, ENV]],
    read_wrt: Callable[[ENV], PARAM_TENSOR],
    write_wrt: Callable[[PARAM_TENSOR, ENV], ENV],
    info: DATA,
    env0: ENV,
) -> tuple[GRADIENT, ENV]:
    parameretrize = reparametrizeInput(lossFn, write_wrt)
    parameretrized: Callable[[DATA, PARAM_TENSOR], tuple[LOSS, ENV]] = (
        lambda x, param: parameretrize(x, env0, param)
    )

    wrt = read_wrt(env0)
    gr, env1 = torch.func.jacrev(parameretrized, argnums=1, has_aux=True)(info, wrt)
    return gr, env1


class _AccumGradient(
    GetGradient[ENV, GRADIENT],
    PutGradient[ENV, GRADIENT],
    GetLoss[ENV, LOSS],
    Protocol[ENV],
):
    pass


def accumGradient(
    dialect: _AccumGradient[ENV],
    step: Callable[[DATA, ENV], ENV],
    foldGradient: ENDOMORPHIC[GRADIENT],
    read_wrt: Callable[[ENV], PARAM_TENSOR],
    write_wrt: Callable[[PARAM_TENSOR, ENV], ENV],
) -> STATEM[DATA, ENV]:
    lossFn = reparameterizeOutput(step, makeLossPair(dialect))

    def endowGrad(x: DATA, env0: ENV) -> ENV:
        gr_old = dialect.getGradient(env0)
        gr, env1 = computeGradient(lossFn, read_wrt, write_wrt, x, env0)
        return dialect.putGradient(foldGradient(gr, gr_old), env1)

    return endowGrad


@dataclass(frozen=True)
class RnnLibrary(Generic[ENV, DATA]):
    activationStep: STATEM[DATA, ENV]
    updatePrediction: STATEM[DATA, ENV]


@dataclass(frozen=True)
class GradientLibrary(Generic[ENV, DATA]):
    rnn: STATEM[DATA, ENV]
    rnnWithLoss: STATEM[DATA, ENV]
    rnnWithParamGradient: Callable[[DATA, ENV], tuple[GRADIENT, ENV]]

    def rnnWithParamGradient_Averaged(
        self, getWeight: Callable[[DATA], float]
    ) -> STATEM[Iterable[DATA], ENV]:
        def weight(data: DATA, pair: tuple[GRADIENT, ENV]) -> tuple[GRADIENT, ENV]:
            prev_gr, prev_env = pair
            weight = getWeight(data)
            curr_gr, curr_env = self.rnnWithParamGradient(data, prev_env)
            return GRADIENT(prev_gr + curr_gr * weight), curr_env

        return lambda xs, env: foldr(weight)(xs, (GRADIENT(torch.tensor(0)), env))


class _LearningLibraryFactory(
    GetGradient[ENV, GRADIENT], PutGradient[ENV, GRADIENT], Protocol[ENV]
):
    pass


def batchGradients(
    dialect: _LearningLibraryFactory[ENV],
    step: STATEM[DATA, ENV],
) -> STATEM[DATA, ENV]:

    def sum_gradients(_: DATA, env: ENV) -> ENV:
        gr_b = dialect.getGradient(env)
        gr = GRADIENT(torch.sum(gr_b, dim=0))
        return dialect.putGradient(gr, env)

    fn_b = torch.vmap(step)
    return collapseF([fn_b, sum_gradients])


@dataclass(frozen=True)
class LearningEffectsLibrary(Generic[ENV]):
    updateGradient: Callable[[GRADIENT, ENV], ENV]
    updateParameter: Callable[[ENV], ENV]


class _ParameterLearn(
    GetParameter[ENV, PARAM],
    PutParameter[ENV, PARAM],
    GetGradient[ENV, GRADIENT],
    PutGradient[ENV, GRADIENT],
    GetHyperParameter[ENV, HPARAM_CO],
    Protocol[ENV, PARAM, HPARAM_CO],
):
    pass


def learningEffectsStandardFactory(
    dialect: _ParameterLearn[ENV, PARAM_TENSOR, HPARAM],
    foldParameter: Callable[[GRADIENT, PARAM_TENSOR, HPARAM], PARAM_TENSOR],
) -> LearningEffectsLibrary[ENV]:

    def updateParameter(env: ENV) -> ENV:
        hp = dialect.getHyperParameter(env)
        param = dialect.getParameter(env)
        grad = dialect.getGradient(env)
        param_ = foldParameter(grad, param, hp)
        return dialect.putParameter(param_, env)

    return LearningEffectsLibrary[ENV](dialect.putGradient, updateParameter)


class _RnnDialect(
    GetActivation[ENV, ACTIV],
    PutActivation[ENV, ACTIV],
    PutParameter[ENV, PARAM],
    GetParameter[ENV, PARAM],
    HasRecurrentWeights[ENV, PARAM, PARAM_R],
    HasReadoutWeights[ENV, PARAM, PARAM_O],
    PutPrediction[ENV, PRED_CON],
    Protocol[ENV, ACTIV, PRED_CON, PARAM, PARAM_R, PARAM_O],
):
    pass


class _RnnDataDialect(
    HasInput[DATA_L, X_L],
    HasPredictionInput[DATA_L, Y_L],
    Protocol[DATA_L, X_L, Y_L, Z_L],
):
    pass


# Smart constructor
def createRnnLibrary(
    dialect: _RnnDialect[ENV, ACTIV, PRED, PARAM, PARAM_R, PARAM_O],
    dataDialect: _RnnDataDialect[DATA, X, Y, Z],
    activationStep: Callable[[X, ACTIV, PARAM_R], ACTIV],
    predictionStep: Callable[[Y, ACTIV, PARAM_O], PRED],
) -> RnnLibrary[ENV, DATA]:
    actv_: STATEM[DATA, ENV] = lambda x, env: activation(
        dialect, dataDialect, activationStep, x, env
    )
    pred_: STATEM[DATA, ENV] = lambda x, env: prediction(
        dialect, dataDialect, predictionStep, x, env
    )
    return RnnLibrary[ENV, DATA](activationStep=actv_, updatePrediction=pred_)


class _Gradient(
    GetPrediction[ENV, PRED_CO],
    GetParameter[ENV, PARAM],
    PutParameter[ENV, PARAM],
    GetGradient[ENV, GRADIENT],
    PutGradient[ENV, GRADIENT],
    GetHyperParameter[ENV, HPARAM_CO],
    GetLoss[ENV, LOSS],
    PutLoss[ENV, LOSS],
    Protocol[ENV, PRED_CO, PARAM, HPARAM_CO],
):
    pass


def offlineLearning(
    dialect: _Gradient[ENV, PRED, PARAM_TENSOR, HPARAM],
    dataDialect: HasLabel[DATA, Z],
    computeLoss: Callable[[Z, PRED], LOSS],
    rnnLibrary: RnnLibrary[ENV, DATA],
) -> GradientLibrary[ENV, Iterable[DATA]]:

    rnn_ = collapseF([rnnLibrary.activationStep, rnnLibrary.updatePrediction])
    rnn = foldr(rnn_)

    loss_: STATEM[DATA, ENV] = lambda x, env: loss(
        dialect, dataDialect, computeLoss, add, x, env
    )
    _step = collapseF([rnn_, loss_])
    rnnWithLoss = foldr(_step)

    def grad_(foldGradient: ENDOMORPHIC[GRADIENT]) -> STATEM[Iterable[DATA], ENV]:
        return accumGradient(
            dialect,
            rnnWithLoss,
            foldGradient,
            dialect.getParameter,
            dialect.putParameter,
        )

    return LearningLibraryStandardParameterFactory(
        dialect, rnn, rnnWithLoss, grad_, foldParameter
    )


class _RtrlDialect(
    GetPrediction[ENV, PRED_CO],
    GetActivation[ENV, ACTIV],
    PutActivation[ENV, ACTIV],
    PutParameter[ENV, PARAM],
    GetParameter[ENV, PARAM],
    PutLoss[ENV, LOSS],
    GetLoss[ENV, LOSS],
    GetGradient[ENV, GRADIENT],
    PutGradient[ENV, GRADIENT],
    GetInfluenceTensor[ENV, INFLUENCETENSOR],
    PutInfluenceTensor[ENV, INFLUENCETENSOR],
    GetHyperParameter[ENV, HPARAM_CO],
    Protocol[ENV, ACTIV, PRED_CO, PARAM, PARAM_R, PARAM_O, HPARAM_CO],
):
    pass


def createRTRLLibrary(
    dialect: _RtrlDialect[ENV, ACTIV_TENSOR, PRED, PARAM, PARAM_R, PARAM_O, HPARAM],
    dataDialect: HasLabel[DATA, Z],
    computeLoss: Callable[[Z, PRED], LOSS],
    foldParameter: Callable[[GRADIENT, PARAM, HPARAM], PARAM],
    rnnLibrary: RnnLibrary[ENV, DATA],
) -> LearningLibrary[ENV, DATA]:

    loss_: STATEM[DATA, ENV] = lambda x, env: loss(
        dialect, dataDialect, computeLoss, lambda _, ls: ls, x, env
    )

    def getInfluenceTensor(data0: DATA, env0: ENV) -> ENV:

        def reparamertrizeActivation(
            d: DATA, a: ACTIV_TENSOR, p: PARAM
        ) -> tuple[ACTIV_TENSOR, ENV]:
            _env1 = dialect.putActivation(a, env0)
            _env2 = dialect.putParameter(p, _env1)
            _env3 = rnnLibrary.activationStep(d, _env2)
            return dialect.getActivation(_env3), _env3

        activOnly: Callable[[DATA, ACTIV_TENSOR, PARAM], ACTIV_TENSOR] = (
            lambda d, a, p: reparamertrizeActivation(d, a, p)[0]
        )

        a0 = dialect.getActivation(env0)
        p0 = dialect.getParameter(env0)
        influenceTensor = dialect.getInfluenceTensor(env0)

        wrt_activ: Callable[[ACTIV_TENSOR], ACTIV_TENSOR] = lambda a: activOnly(
            data0, a, p0
        )

        # I take the jvp bc 'wrt_activ' could compute a jacobian, so I'd like to 1) do a hvp 2) forward over reverse is efficient: https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
        immediateJacobianInfluenceProduct: torch.Tensor = jacobian_matrix_product(
            wrt_activ, a0, influenceTensor
        )
        # performance below really depends on the problem. For RTRL, I should use jacrev. For OHO, jacfwd. Will make this configurable later.
        immediateInfluence, env = torch.func.jacfwd(
            reparamertrizeActivation, argnums=2, has_aux=True
        )(data0, a0, p0)

        influenceTensor_: INFLUENCETENSOR = INFLUENCETENSOR(
            immediateJacobianInfluenceProduct + immediateInfluence
        )
        env_ = dialect.putInfluenceTensor(influenceTensor_, env)
        return env_

    #! We do not accumulate loss in online pass
    readoutFn = collapseF(iter([rnnLibrary.updatePrediction, loss_]))
    readoutLoss = reparameterizeOutput(readoutFn, makeLossPair(dialect))

    def creditAssignment(
        foldGradient: ENDOMORPHIC[GRADIENT], data0: DATA, env0: ENV
    ) -> ENV:
        immediateCreditAssignment, env1 = computeGradient(
            readoutLoss, dialect.getActivation, dialect.putActivation, data0, env0
        )
        influenceTensor = dialect.getInfluenceTensor(env1)
        credit_gr = GRADIENT(immediateCreditAssignment @ influenceTensor)
        return fmapState(
            dialect.getGradient,
            dialect.putGradient,
            lambda gr: foldGradient(gr, credit_gr),
        )(data0, env1)

    def grad_(foldGradient: ENDOMORPHIC[GRADIENT]) -> STATEM[DATA, ENV]:
        ca = lambda d, env: creditAssignment(foldGradient, d, env)
        return collapseF(iter([getInfluenceTensor, ca]))

    def optimize_(env: ENV) -> ENV:
        hp = dialect.getHyperParameter(env)
        param = dialect.getParameter(env)
        grad = dialect.getGradient(env)
        param_ = foldParameter(grad, param, hp)
        return dialect.putParameter(param_, env)

    return LearningLibrary[ENV, DATA](
        rnn=collapseF(iter([rnnLibrary.activationStep, rnnLibrary.updatePrediction])),
        updateLoss=collapseF(
            iter([rnnLibrary.activationStep, rnnLibrary.updatePrediction, loss_])
        ),
        rnnWithParamGradient=grad_,
        updateParameter=optimize_,
    )


def createRTRLLibrary(
    dialect: _RtrlDialect[ENV, ACTIV_TENSOR, PRED, PARAM, PARAM_R, PARAM_O, HPARAM],
    dataDialect: HasLabel[DATA, Z],
    computeLoss: Callable[[Z, PRED], LOSS],
    foldParameter: Callable[[GRADIENT, PARAM, HPARAM], PARAM],
    getInfluenceTensor: STATEM[DATA, ENV],
    rnnLibrary: RnnLibrary[ENV, DATA],
) -> LearningLibrary[ENV, DATA]:

    #! We do not accumulate loss in online pass
    loss_: STATEM[DATA, ENV] = lambda x, env: loss(
        dialect, dataDialect, computeLoss, lambda _, ls: ls, x, env
    )

    readoutFn = collapseF(iter([rnnLibrary.updatePrediction, loss_]))
    readoutLoss = reparameterizeOutput(readoutFn, makeLossPair(dialect))

    def creditAssignment(
        foldGradient: ENDOMORPHIC[GRADIENT], data0: DATA, env0: ENV
    ) -> ENV:
        immediateCreditAssignment, env1 = computeGradient(
            readoutLoss, dialect.getActivation, dialect.putActivation, data0, env0
        )
        influenceTensor = dialect.getInfluenceTensor(env1)
        credit_gr = GRADIENT(immediateCreditAssignment @ influenceTensor)
        return fmapState(
            dialect.getGradient,
            dialect.putGradient,
            lambda gr: foldGradient(gr, credit_gr),
        )(data0, env1)

    def grad_(foldGradient: ENDOMORPHIC[GRADIENT]) -> STATEM[DATA, ENV]:
        ca = lambda d, env: creditAssignment(foldGradient, d, env)
        return collapseF(iter([getInfluenceTensor, ca]))

    rnn = collapseF(iter([rnnLibrary.activationStep, rnnLibrary.updatePrediction]))
    rnnWithLoss = collapseF(
        iter([rnnLibrary.activationStep, rnnLibrary.updatePrediction, loss_])
    )

    return LearningLibrary[ENV, DATA](
        rnn=collapseF(iter([rnnLibrary.activationStep, rnnLibrary.updatePrediction])),
        updateLoss=collapseF(
            iter([rnnLibrary.activationStep, rnnLibrary.updatePrediction, loss_])
        ),
        rnnWithParamGradient=grad_,
        updateParameter=optimize_,
    )


# creditAssignment = accumGradient(
#     dialect, readoutFn, lambda _, b: b, dialect.getGradient, dialect.putGradient
# )
# lossFn = collapseF(
#     iter(
#         [
#             getInfluenceTensor,
#             rnnLibrary.updatePrediction,
#             rnnLibrary.foldLoss(lambda _, ls: ls),
#         ]
#     )
# )
# todo

# def getCreditAssignment(data0: DATA, env0: ENV) -> ENV:
#     def reparamertrizeLoss(d: DATA, env: ENV) -> tuple[LOSS, ENV]:
#         env_ = rnnLibrary.foldLoss(lambda _, ls: ls)(
#             d, env
#         )
#         return dialect.getLoss(env_), env_


#!!!!!!!! WILLIAM REFER TO THIS: turn this into a forward passing guys as well
# def getGradient(
#     dialect: _Gradient[ENV, PARAM],
#     lossFn: Callable[[DATA, ENV], LOSS],
#     foldGradient: ENDOMORPHIC[GRADIENT],
#     info: DATA,
#     env: ENV,
# ) -> ENV:
#     parameretrize = reparametrizeInput(lossFn, dialect.putParameter)
#     p = dialect.getParameter(env)
#     parameretrized: Callable[[DATA, PARAM], LOSS] = lambda x, param: parameretrize(
#         x, env, param
#     )
#     gr = torch.func.jacrev(parameretrized, argnums=1)(info, p)
#     return dialect.putGradient(foldGradient(gr, dialect.getGradient(env)), env)


# @dataclass(frozen=True)
# class GradientLibrary(Generic[ENV, DATA, ACTIV, PRED]):
#     offlineGradient: Callable[[Iterator[DATA], ENV], ENV]
#     averageGradient: Callable[[int], Callable[[Iterator[DATA], ENV], ENV]]

# lossFn = collapseF(iter([actv_, pred_, loss_]))
#     totalLossFn = reparameterizeOutput(foldr(lossFn), dialect.getLoss)
#     grad_: Callable[[Iterator[DATA], ENV], ENV] = lambda xs, env: getGradient(dialect, totalLossFn, xs, env)

# big idea, don't need to actually combine grad with the rest,
# reparametertize input takes tuple, rewrite my actual steps into take tuple as second arg
# avg loss
# to make online, you do need to combine grad with the rest.
# to reparaemeterize x -> env -> env, you can curry away with env0. THis makes sense because the inside function shouldn't depend on the state of the outside function.

# two things I'm confused about
# 1. how to average gradients
# 2. should I average losses? A: NO. I will do the averaging, auxillarily
# 3. how to do online learning, and how does averaging it work


# def getAverageGradient(gradientFn: Callable[[ENV, DATA], tuple[ENV, GRADIENT]],
#                 env0: ENV,
#                 infos: Iterator[DATA]) -> GRADIENT:
#     # if get empty list, means env should stay as it is, so "avg gradient" is zero
#     def fold(pair: tuple[ENV, GRADIENT, int], info: DATA) -> tuple[ENV, GRADIENT, int]:
#         env, running_gr, running_length = pair
#         env_, gr_ = gradientFn(env, info)
#         return env_, running_gr + gr_, running_length + 1


# -- offlineAverageGradient ::
# --   ( HasActivation env Activation,
# --     HasParameter env param,
# --     HasReccurentWeights env param Parameter,
# --     HasReadoutWeights env param Parameter,
# --     HasLossFn env,
# --     HasTrainOutput info OutputFeature,
# --     HasTrainInput info InputFeature,
# --     HasReadoutFn env,
# --     HasActivationFn env
# --   ) =>
# --   env ->
# --   [[info]] ->
# --   Gradient
# -- offlineAverageGradient env xs = avg . foldr ((<>) . offlineGradient env) mempty $ xs
# --   where
# --     n = fromIntegral $ length xs
# --     avg = Gradient . V.map (M.Sum . (/ n) . M.getSum) . _gradient


# -- offlineTrainLoop ::
# --   ( HasParameter env Parameter,
# --     HasHyperParameter env HyperParameter,
# --     HasGradientOptimizer env,
# --     HasActivation env Activation,
# --     HasLossFn env,
# --     HasTrainOutput info OutputFeature,
# --     HasTrainInput info InputFeature,
# --     HasReccurentWeights env Parameter Parameter,
# --     HasReadoutWeights env Parameter Parameter,
# --     HasReadoutFn env,
# --     HasActivationFn env
# --   ) =>
# --   env ->
# --   [[[info]]] ->
# --   env
# -- offlineTrainLoop = foldrfl $ liftA2 (.) putParameter $ updateParameter offlineAverageGradient -- vmap here before folding over dataloader


# def averageGradient(  xs: Iterator[X]
#                     , envWithGrad: tuple[ENV, torch.Tensor]
#                     , foldable: Callable[[Iterator[X], tuple[ENV, torch.Tensor]], tuple[ENV, torch.Tensor]]) -> tuple[ENV, torch.Tensor]:
#     env, grads = foldable(xs, envWithGrad)
#     grads_avg = grads / len(xs)
#     return env, grads_avg

# def splitTimeSeries(chunkSize: int, bptt: Callable[[Iterator[Iterator[X]], ENV], ENV]) -> Callable[[Iterator[X], ENV], ENV]:
#     def splitTimeSeries_(xs: Iterator[X], env: ENV) -> ENV:
#         xs_ = itertools.batched(xs, chunkSize)  #! Inefficient, always going to loop twice.
#         return bptt(xs_, env)
#     return splitTimeSeries_


# def rnnStep(  config: RnnConfig
#             , x: torch.Tensor
#             , a: torch.Tensor
#             , w_rec: torch.Tensor) -> torch.Tensor:
#     w_rec = torch.reshape(w_rec, (config.n_h, config.n_h+config.n_in+1))
#     return (1 - config.alpha) * a + config.alpha * config.activation(w_rec @ torch.cat((x, a, torch.tensor([1.0]))))

# def rnnReadout(   config: RnnConfig
#                 , a: torch.Tensor
#                 , w_out: torch.Tensor) -> torch.Tensor:
#     w_out = torch.reshape(w_out, (config.n_out, config.n_h+1))
#     return w_out @ torch.cat((a, torch.tensor([1.0])))


# def rnnSplitParameters(parameters: torch.Tensor, config: RnnConfig) -> tuple[torch.Tensor, torch.Tensor]:
#     w_rec, w_out = torch.split(parameters, [config.n_h*(config.n_h+config.n_in+1), config.n_out*(config.n_h+1)])
#     return w_rec, w_out


# def rnnActivation_Vanilla(t: Union[HasActivation[ENV, torch.Tensor], HasParameter[ENV, RNN_PARAM]]) -> Callable[[torch.Tensor, ENV], ENV]:
#     def rnnActivation_Vanilla__(x: torch.Tensor, env: ENV) -> ENV:
#         a = t.getActivation(env)
#         parameters, config = t.getParameter(env)
#         w_rec, _ = rnnSplitParameters(parameters, config)
#         a_ = rnnStep(config, x, a, w_rec)
#         return t.putActivation(a_, env)
#     return rnnActivation_Vanilla__


# def rnnPrediction_Vanilla(algebra: Union[HasActivation[ENV, torch.Tensor], HasParameter[ENV, RNN_PARAM]]) -> Callable[[ENV], torch.Tensor]:
#     def rnnPrediction_Vanilla_(env: ENV) -> torch.Tensor:
#         a = algebra.getActivation(env)
#         parameters, config = algebra.getParameter(env)
#         _, w_out = rnnSplitParameters(parameters, config)
#         return rnnReadout(config, a, w_out)
#     return rnnPrediction_Vanilla_


# def truncatedRNN_Vanilla( criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
#                         , algebra: Union[
#                             HasActivation[ENV, torch.Tensor]
#                             , HasParameter[ENV, RNN_PARAM]
#                             , HasTrainingInput[DATA, torch.Tensor]
#                             , HasTrainingLabel[DATA, torch.Tensor]]):
#     actv = rnnActivation_Vanilla(algebra)
#     predict = rnnPrediction_Vanilla(algebra)
#     def step(data: DATA, envWithLoss: tuple[ENV, torch.Tensor]) -> tuple[ENV, torch.Tensor]:
#         x, y = algebra.getTrainingInput(data), algebra.getTrainingLabel(data)
#         env, loss = envWithLoss
#         env_ = actv(x, env)
#         return env_, loss + criterion(predict(env_), y)
#     return foldr(step)


# def torchGradient(extractTorchParam: Callable[[P], torch.Tensor]):
#     def torchGradient_(algebra: Union[HasParameter[ENV, P]]) -> Callable[[ENV, torch.Tensor], torch.Tensor]:
#         def torchGradient__(lossGraph: torch.Tensor, env: ENV) -> torch.Tensor:
#             p = algebra.getParameter(env)
#             param = extractTorchParam(p)
#             grad = jacobian(lossGraph, param)
#             return grad
#         return torchGradient__
#     return torchGradient_


# def torchUpdateParam( envWithGrad: tuple[ENV, torch.Tensor]
#                     , optimizer: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
#                     , extractTorchParam: Callable[[P], torch.Tensor]
#                     , torchToParam: Callable[[torch.Tensor, P], P]
#                     , algebra: Union[HasParameter[ENV, P]]) -> ENV:
#     env, grad = envWithGrad
#     p = algebra.getParameter(env)
#     param = extractTorchParam(p)
#     param_ = optimizer(param, grad)
#     p_ = torchToParam(param_, p)
#     return algebra.putParameter(p_, env)


# def efficientBPTTGradient(rnnLoss: Callable[[X, tuple[ENV, torch.Tensor]], tuple[ENV, torch.Tensor]]
#                         , algebra: Union[
#                             HasActivation[ENV, torch.Tensor]
#                             , HasParameter[ENV, RNN_PARAM]]):

#     def step(data: X, envWithGrad: tuple[ENV, torch.Tensor]) -> tuple[ENV, torch.Tensor]:  # TODO: stop assume loss starts at zero after every update
#         env, grad = envWithGrad
#         env_, loss = rnnLoss(data, (env, torch.tensor(0.0)))
#         grad_ = torchGradient(fst)(algebra)(loss, env_)
#         return env_, grad + grad_

#     return lambda xs, envWithGrad: averageGradient(xs, envWithGrad, foldr(step))  # TODO: stop assuming average later on

# def efficientBPTT(optimizer: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
#                 , rnnLoss: Callable[[X, tuple[ENV, torch.Tensor]], tuple[ENV, torch.Tensor]]
#                 , algebra: Union[
#                     HasActivation[ENV, torch.Tensor]
#                     , HasParameter[ENV, RNN_PARAM]]):
#     def update(xs: Iterator[X]
#             , env: ENV) -> ENV:
#         parameters, _ = algebra.getParameter(env)
#         envWithGrad = (env, torch.zeros_like(parameters))
#         env_, grad_avg = efficientBPTTGradient(rnnLoss, algebra)(xs, envWithGrad)
#         return torchUpdateParam((env_, grad_avg), optimizer, fst, lambda p_, rnnP: (p_, snd(rnnP)), algebra)
#     return update


# def efficientBPTT_Vanilla(optimizer: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
#                         , criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
#                         , algebra: Union[
#                             HasActivation[ENV, torch.Tensor]
#                             , HasParameter[ENV, RNN_PARAM]
#                             , HasTrainingInput[DATA, torch.Tensor]
#                             , HasTrainingLabel[DATA, torch.Tensor]]):
#     rnnLoss = truncatedRNN_Vanilla(criterion, algebra)
#     bptt = efficientBPTT(optimizer, rnnLoss, algebra)
#     return bptt

# def efficientBPTT_Vanilla_Full(optimizer: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
#                             , criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
#                             , algebra: Union[
#                             HasActivation[ENV, torch.Tensor]
#                             , HasParameter[ENV, RNN_PARAM]
#                             , HasTrainingInput[DATA, torch.Tensor]
#                             , HasTrainingLabel[DATA, torch.Tensor]]):
#     rnnLoss = truncatedRNN_Vanilla(criterion, algebra)
#     bptt = efficientBPTT(optimizer, rnnLoss, algebra)
#     def bpttFull(xs: list[DATA], env: ENV) -> ENV:
#         return splitTimeSeries(len(xs), bptt)(xs, env)
#     return bpttFull


# # as long as I dont have the EVERYTHING is a multilayer RNN set up, I will have to manually code prediction separetly from the training.
# def rnnPrediction_Vanilla(algebra: Union[
#                             HasActivation[ENV, torch.Tensor]
#                             , HasParameter[ENV, RNN_PARAM]
#                             , HasTrainingInput[DATA, torch.Tensor]]):
#     actv = rnnActivation_Vanilla(algebra)
#     predict = rnnPrediction_Vanilla(algebra)

#     def step(data: DATA, env: tuple[ENV]) -> tuple[ENV]:
#         x = algebra.getTrainingInput(data)
#         env_ = actv(x, env)
#         return env_

#     def readouts(xs: Iterator[DATA], env: ENV) -> torch.Tensor:
#         envs = scan0(step, env, xs)
#         return torch.stack(list(map(predict, envs)))

#     return readouts


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
