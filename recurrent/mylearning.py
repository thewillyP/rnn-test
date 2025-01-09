from dataclasses import dataclass
from typing import Callable, Generic, TypeVar, Protocol, Iterable, cast
import torch
from recurrent.myrecords import RfloConfig
from recurrent.objectalgebra.typeclasses import (
    GetActivation,
    GetHyperParameter,
    GetInfluenceTensor,
    GetParameter,
    GetRfloConfig,
    HasReadoutWeights,
    HasInput,
    HasPredictionInput,
    HasLabel,
    HasRecurrentWeights,
    PutActivation,
    PutInfluenceTensor,
    PutParameter,
)

from recurrent.mytypes import *
from operator import add
from recurrent.monad import (
    ReaderState,
    State,
    Unit,
    foldM_,
    runReader,
    toReader,
    reader,
    traverse,
)


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
ENV_CON = TypeVar("ENV_CON", contravariant=True)
PARAM_CON = TypeVar("PARAM_CON", contravariant=True)

PARAM_TENSOR = TypeVar("PARAM_TENSOR", bound=torch.Tensor)
ACTIV_TENSOR = TypeVar("ACTIV_TENSOR", bound=torch.Tensor)

DATA_NEW = TypeVar("DATA_NEW")

T_CON = TypeVar("T_CON", contravariant=True)
T_CO = TypeVar("T_CO", covariant=True)


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
) -> ReaderState[DATA, ENV, Unit]:
    def update(pair: tuple[DATA, ENV]) -> ENV:
        data, env = pair
        a = dialect.getActivation(env)
        x = dataDialect.getInput(data)
        w = dialect.getRecurrentWeights(dialect.getParameter(env), env)
        return dialect.putActivation(step(x, a, w), env)

    return (
        ReaderState[DATA, ENV, Unit]
        .ask_get()
        .fmap(update)
        .bind(ReaderState[DATA, ENV, Unit].put)
    )


class _Prediction(
    GetActivation[ENV_CON, ACTIV_CO],
    GetParameter[ENV_CON, PARAM],
    HasReadoutWeights[ENV_CON, PARAM, PARAM_T],
    Protocol[ENV_CON, ACTIV_CO, PARAM, PARAM_T],
):
    pass


class _PredictionData(HasPredictionInput[DATA_L, X_L], Protocol[DATA_L, X_L]):
    pass


def prediction(
    dialect: _Prediction[ENV, ACTIV, PARAM, PARAM_T],
    dataDialect: _PredictionData[DATA, X],
    step: Callable[[X, ACTIV, PARAM_T], PRED],
) -> ReaderState[DATA, ENV, PRED]:
    def update(pair: tuple[DATA, ENV]) -> PRED:
        data, env = pair
        x = dataDialect.getPredictionInput(data)
        a = dialect.getActivation(env)
        w_out = dialect.getReadoutWeights(dialect.getParameter(env), env)
        return step(x, a, w_out)

    return ReaderState[DATA, ENV, tuple[DATA, ENV]].ask_get().fmap(update)


def loss(
    dataDialect: HasLabel[DATA, Y],
    computeLoss: Callable[[Y, PRED], LOSS],
) -> Callable[[PRED], ReaderState[DATA, ENV, LOSS]]:
    def bindLoss(pred: PRED) -> ReaderState[DATA, ENV, LOSS]:
        return (
            ReaderState[DATA, ENV, DATA]
            .ask()
            .fmap(lambda data: computeLoss(dataDialect.getLabel(data), pred))
        )

    return bindLoss


def computeGradient(
    lossM: ReaderState[DATA, ENV, LOSS],
    read_wrt: Callable[[ENV], PARAM_TENSOR],
    write_wrt: Callable[[PARAM_TENSOR, ENV], ENV],
) -> ReaderState[DATA, ENV, GRADIENT]:
    parameretrize = reparametrizeInput(lossM.run, write_wrt)

    def parameretrized(env: ENV) -> Callable[[DATA, PARAM_TENSOR], tuple[LOSS, ENV]]:
        return lambda x, param: parameretrize(x, env, param)

    def getGradient(data: DATA, env: ENV) -> tuple[GRADIENT, ENV]:
        gr, env_new = torch.func.jacrev(parameretrized(env), argnums=1)(
            data, read_wrt(env)
        )
        return GRADIENT(gr), env_new

    return reader(getGradient)


@dataclass(frozen=True)
class RnnLibrary(Generic[ENV, PRED, DATA]):
    activationStep: ReaderState[DATA, ENV, Unit]
    updatePrediction: ReaderState[DATA, ENV, PRED]

    def rnnStep(self) -> ReaderState[DATA, ENV, PRED]:
        return self.activationStep.bind(lambda _: self.updatePrediction)


@dataclass(frozen=True)
class GradientLibrary(Generic[ENV, PRED, DATA]):
    rnn: ReaderState[DATA, ENV, PRED]
    rnnWithLoss: ReaderState[DATA, ENV, LOSS]
    rnnWithParamGradient: ReaderState[DATA, ENV, GRADIENT]

    def rnnWithParamGradient_Averaged(
        self, getWeight: Callable[[DATA], float]
    ) -> ReaderState[Iterable[DATA], ENV, GRADIENT]:
        def weight(data: DATA, gr: GRADIENT) -> State[ENV, GRADIENT]:
            return runReader(self.rnnWithParamGradient)(data).fmap(
                lambda gr_: GRADIENT(gr + gr_ * getWeight(data))
            )

        return toReader(foldM_(weight, GRADIENT(torch.tensor(0))))


def batchGradients(
    step: ReaderState[DATA, ENV, GRADIENT],
) -> ReaderState[DATA, ENV, GRADIENT]:
    return ReaderState[DATA, ENV, GRADIENT](torch.vmap(step.run)).fmap(
        lambda gr: GRADIENT(torch.sum(gr, dim=0))
    )


@dataclass(frozen=True)
class LearningEffectsLibrary(Generic[ENV, DATA]):
    updateParameter: Callable[[GRADIENT], ReaderState[DATA, ENV, Unit]]


class _ParameterLearn(
    GetParameter[ENV, PARAM],
    PutParameter[ENV, PARAM],
    GetHyperParameter[ENV, HPARAM_CO],
    Protocol[ENV, PARAM, HPARAM_CO],
):
    pass


def learningEffectsStandardFactory(
    dialect: _ParameterLearn[ENV, PARAM_TENSOR, HPARAM],
    foldParameter: Callable[[GRADIENT, PARAM_TENSOR, HPARAM], PARAM_TENSOR],
) -> LearningEffectsLibrary[ENV, DATA]:
    def updateParameter(gr: GRADIENT) -> ReaderState[DATA, ENV, Unit]:
        return (
            ReaderState[DATA, ENV, ENV]
            .get()
            .fmap(
                lambda env: dialect.putParameter(
                    foldParameter(
                        gr, dialect.getParameter(env), dialect.getHyperParameter(env)
                    ),
                    env,
                )
            )
            .bind(ReaderState[DATA, ENV, Unit].put)
        )

    return LearningEffectsLibrary[ENV, DATA](updateParameter)


class _RnnDialect(
    GetActivation[ENV, ACTIV],
    PutActivation[ENV, ACTIV],
    PutParameter[ENV, PARAM],
    GetParameter[ENV, PARAM],
    HasRecurrentWeights[ENV, PARAM, PARAM_R],
    HasReadoutWeights[ENV, PARAM, PARAM_O],
    Protocol[ENV, ACTIV, PARAM, PARAM_R, PARAM_O],
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
    dialect: _RnnDialect[ENV, ACTIV, PARAM, PARAM_R, PARAM_O],
    dataDialect: _RnnDataDialect[DATA, X, Y, Z],
    activationStep: Callable[[X, ACTIV, PARAM_R], ACTIV],
    predictionStep: Callable[[Y, ACTIV, PARAM_O], PRED],
) -> RnnLibrary[ENV, PRED, DATA]:
    actv_ = activation(dialect, dataDialect, activationStep)
    pred_ = prediction(dialect, dataDialect, predictionStep)
    return RnnLibrary[ENV, PRED, DATA](activationStep=actv_, updatePrediction=pred_)


class _Gradient(
    GetParameter[ENV, PARAM],
    PutParameter[ENV, PARAM],
    GetHyperParameter[ENV, HPARAM_CO],
    Protocol[ENV, PRED_CO, PARAM, HPARAM_CO],
):
    pass


def offlineLearning(
    dialect: _Gradient[ENV, PRED, PARAM_TENSOR, HPARAM],
    dataDialect: HasLabel[DATA, Z],
    computeLoss: Callable[[Z, PRED], LOSS],
    rnnLibrary: RnnLibrary[ENV, PRED, DATA],
) -> GradientLibrary[ENV, Iterable[PRED], Iterable[DATA]]:

    rnnWithPred = toReader(traverse(runReader(rnnLibrary.rnnStep())))

    rnn_with_loss = rnnLibrary.rnnStep().bind(loss(dataDialect, computeLoss))

    def accum_loss(d: DATA, accum: LOSS) -> State[ENV, LOSS]:
        return runReader(rnn_with_loss)(d).fmap(lambda l: LOSS(accum + l))

    rnnWithLoss = toReader(foldM_(accum_loss, LOSS(torch.tensor(0))))

    rnnWithGrad = computeGradient(
        rnnWithLoss, dialect.getParameter, dialect.putParameter
    )

    return GradientLibrary[ENV, Iterable[PRED], Iterable[DATA]](
        rnn=rnnWithPred,
        rnnWithLoss=rnnWithLoss,
        rnnWithParamGradient=rnnWithGrad,
    )


class _EndowPastFacingGradient(
    GetActivation[ENV, ACTIV],
    PutActivation[ENV, ACTIV],
    GetParameter[ENV, PARAM],
    PutParameter[ENV, PARAM],
    GetInfluenceTensor[ENV, INFLUENCETENSOR],
    Protocol[ENV, ACTIV, PARAM],
):
    pass


def endowPastFacingGradient(
    dialect: _EndowPastFacingGradient[ENV, ACTIV_TENSOR, PARAM],
    dataDialect: HasLabel[DATA, Z],
    computeLoss: Callable[[Z, PRED], LOSS],
    getInfluenceTensorFactory: Callable[
        [
            ACTIV_TENSOR,
            PARAM,
            INFLUENCETENSOR,
            ENV,
            Callable[[ACTIV_TENSOR, PARAM], tuple[ACTIV_TENSOR, ENV]],
        ],
        ENV,
    ],
    rnnLibrary: RnnLibrary[ENV, PRED, DATA],
) -> GradientLibrary[ENV, PRED, DATA]:

    readout_loss = rnnLibrary.updatePrediction.bind(loss(dataDialect, computeLoss))

    immediateCreditAssignmentM = computeGradient(
        readout_loss, dialect.getActivation, dialect.putActivation
    )
    creditAssignment = (
        ReaderState[DATA, ENV, INFLUENCETENSOR]
        .get()
        .fmap(dialect.getInfluenceTensor)
        .bind(
            lambda influenceTensor: immediateCreditAssignmentM.fmap(
                lambda immediateCreditAssignment: GRADIENT(
                    immediateCreditAssignment @ influenceTensor
                )
            )
        )
    )

    def buildInfluenceTensor(data0: DATA, env0: ENV) -> tuple[Unit, ENV]:
        def reparametrizeActivation(
            a: ACTIV_TENSOR, p: PARAM
        ) -> tuple[ACTIV_TENSOR, ENV]:
            doActiv = (
                ReaderState[DATA, ENV, ENV]
                .get()
                .fmap(lambda env: dialect.putActivation(a, env))
                .fmap(lambda env: dialect.putParameter(p, env))
                .bind(ReaderState[DATA, ENV, ENV].put)
                .then(rnnLibrary.activationStep)
                .then(ReaderState[DATA, ENV, ENV].get())
                .fmap(dialect.getActivation)
            )
            return doActiv.run(data0, env0)

        a0 = dialect.getActivation(env0)
        p0 = dialect.getParameter(env0)
        influenceTensor = dialect.getInfluenceTensor(env0)

        return Unit(), getInfluenceTensorFactory(
            a0, p0, influenceTensor, env0, reparametrizeActivation
        )

    gradient = reader(buildInfluenceTensor).then(creditAssignment)

    return GradientLibrary[ENV, PRED, DATA](
        rnn=rnnLibrary.rnnStep(),
        rnnWithLoss=rnnLibrary.rnnStep().bind(loss(dataDialect, computeLoss)),
        rnnWithParamGradient=gradient,
    )


class _RtrlDialect(
    GetActivation[ENV, ACTIV],
    PutActivation[ENV, ACTIV],
    PutParameter[ENV, PARAM],
    GetParameter[ENV, PARAM],
    GetInfluenceTensor[ENV, INFLUENCETENSOR],
    PutInfluenceTensor[ENV, INFLUENCETENSOR],
    Protocol[ENV, ACTIV, PARAM],
):
    pass


def RTRL(
    dialect: _RtrlDialect[ENV, ACTIV_TENSOR, PARAM],
    dataDialect: HasLabel[DATA, Z],
    computeLoss: Callable[[Z, PRED], LOSS],
    rnnLibrary: RnnLibrary[ENV, PRED, DATA],
) -> GradientLibrary[ENV, PRED, DATA]:

    def getInfluenceTensor(
        a0: ACTIV,
        p0: PARAM,
        influenceTensor: INFLUENCETENSOR,
        _: ENV,
        reparametrizeActivation: Callable[
            [ACTIV_TENSOR, PARAM], tuple[ACTIV_TENSOR, ENV]
        ],
    ) -> ENV:
        # I take the jvp bc 'wrt_activ' could compute a jacobian, so I'd like to 1) do a hvp 2) forward over reverse is efficient: https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
        immediateJacobianInfluenceProduct: torch.Tensor = jacobian_matrix_product(
            lambda a: reparametrizeActivation(a, p0)[0],
            a0,
            influenceTensor,
        )
        # performance below really depends on the problem. For RTRL, I should use jacrev. For OHO, jacfwd. Will make this configurable later.
        immediateInfluence, env = torch.func.jacfwd(
            reparametrizeActivation, argnums=1, has_aux=True
        )(a0, p0)

        influenceTensor_ = INFLUENCETENSOR(
            immediateJacobianInfluenceProduct + immediateInfluence
        )
        return dialect.putInfluenceTensor(influenceTensor_, env)

    return endowPastFacingGradient(
        dialect, dataDialect, computeLoss, getInfluenceTensor, rnnLibrary
    )


class _RfloDialect(
    _RtrlDialect[ENV, ACTIV, PARAM],
    GetRfloConfig[ENV],
    Protocol[ENV, ACTIV, PARAM],
):
    pass


def RFLO(
    dialect: _RfloDialect[ENV, ACTIV_TENSOR, PARAM],
    dataDialect: HasLabel[DATA, Z],
    computeLoss: Callable[[Z, PRED], LOSS],
    rnnLibrary: RnnLibrary[ENV, PRED, DATA],
) -> GradientLibrary[ENV, PRED, DATA]:

    def getInfluenceTensor(
        a0: ACTIV,
        p0: PARAM,
        influenceTensor: INFLUENCETENSOR,
        env0: ENV,
        reparametrizeActivation: Callable[
            [ACTIV_TENSOR, PARAM], tuple[ACTIV_TENSOR, ENV]
        ],
    ) -> ENV:
        # python doesn't understand closures, so I need to explictly tell it
        alpha = dialect.getRfloConfig(env0).rflo_alpha

        immediateInfluence, env = torch.func.jacrev(
            reparametrizeActivation, argnums=1, has_aux=True
        )(a0, p0)

        influenceTensor_ = INFLUENCETENSOR(
            (1 - alpha) * influenceTensor + alpha * immediateInfluence
        )
        return dialect.putInfluenceTensor(influenceTensor_, env)

    return endowPastFacingGradient(
        dialect, dataDialect, computeLoss, getInfluenceTensor, rnnLibrary
    )


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
