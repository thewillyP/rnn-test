from dataclasses import dataclass
from typing import Callable, Generic, TypeVar, Iterator, Protocol, Type
import torch
from myobjectalgebra import HasActivation, HasGradient, HasLoss, HasParameter, HasPrediction, HasReadoutWeights, HasInput, HasPredictionInput, HasLabel, HasRecurrentWeights, HasHyperParameter
from myfunc import collapseF, flip, foldr, scan0, fst, snd, liftA2Compose, foldl
import itertools
from mytypes import *
from operator import add

ACTIV = TypeVar('ACTIV')
PRED = TypeVar('PRED')
PARAM = TypeVar('PARAM')
HPARAM = TypeVar('HPARAM')
PARAM_T = TypeVar('PARAM_T', covariant=True)
PARAM_R = TypeVar('PARAM_R', covariant=True)
PARAM_O = TypeVar('PARAM_O', covariant=True)

DATA_L = TypeVar('DATA_L', contravariant=True)
X_L = TypeVar('X_L', covariant=True)
Y_L = TypeVar('Y_L', covariant=True)
Z_L = TypeVar('Z_L', covariant=True)


class _Activation(HasActivation[ENV, ACTIV], HasParameter[ENV, PARAM], HasRecurrentWeights[ENV, PARAM, PARAM_T], Protocol[ENV, ACTIV, PARAM, PARAM_T]):
    pass

class _ActivationData(HasInput[DATA_L, X_L], Protocol[DATA_L, X_L]):
    pass

def activation(dialect: _Activation[ENV, ACTIV, PARAM, PARAM_T],
            dataDialect: _ActivationData[DATA, X],
            step: Callable[[X, ACTIV, PARAM_T], ACTIV],
            info: DATA,
            env: ENV) -> ENV:
    a = dialect.getActivation(env)
    x = dataDialect.getInput(info)
    w_rec = dialect.getRecurrentWeights(dialect.getParameter(env), env)
    a_ = step(x, a, w_rec)
    return dialect.putActivation(a_, env)


class _Prediction(HasActivation[ENV, ACTIV], HasParameter[ENV, PARAM], HasReadoutWeights[ENV, PARAM, PARAM_T], HasPrediction[ENV, PRED], Protocol[ENV, ACTIV, PARAM, PARAM_T, PRED]):
    pass

class _PredictionData(HasPredictionInput[DATA_L, X_L], Protocol[DATA_L, X_L]):
    pass

def prediction(dialect: _Prediction[ENV, ACTIV, PARAM, PARAM_T, PRED],
                dataDialect: _PredictionData[DATA, X],
                step: Callable[[X, ACTIV, PARAM_T], PRED],  # oho readout doesnt use hyperparameter as its readout though...
                info: DATA,
                env: ENV) -> ENV:
    a = dialect.getActivation(env)
    x = dataDialect.getPredictionInput(info)
    w_out = dialect.getReadoutWeights(dialect.getParameter(env), env)
    pred = step(x, a, w_out)
    return dialect.putPrediction(pred, env)

class _Loss(HasLoss[ENV, LOSS], HasPrediction[ENV, PRED], Protocol[ENV, PRED]):
    pass

def loss(dialect: _Loss[ENV, PRED],
        dataDialect: HasLabel[DATA, Y],
        computeLoss: Callable[[Y, PRED], LOSS],
        lossStep: Callable[[LOSS, LOSS], LOSS],
        info: DATA,
        env: ENV) -> ENV:
    y = dataDialect.getLabel(info)
    pred = dialect.getPrediction(env)
    l = lossStep(dialect.getLoss(env), computeLoss(y, pred))
    return dialect.putLoss(l, env)

# literally State get
def reparameterizeOutput(step: Callable[[DATA, ENV], ENV], read: Callable[[ENV], X]) -> Callable[[DATA, ENV], X]:
    def reparametrized(info: DATA, env: ENV) -> X:
        env_ = step(info, env)
        return read(env_)
    return reparametrized

# literally State put
def reparametrizeInput(step: Callable[[DATA, ENV], Z], writer: Callable[[X, ENV], ENV]) -> Callable[[DATA, ENV, X], Z]:
    def reparametrized(info: DATA, env: ENV, x: X) -> Z:
        env_ = writer(x, env)
        return step(info, env_)
    return reparametrized





# def avgGradient(dialect: HasGradient[ENV, GRADIENT],
#                 gradStep: Callable[[GRADIENT, GRADIENT], GRADIENT],
#                 info: DATA,
#                 env: ENV) -> ENV:
#     gr = dialect.getGradient(env)
#     gr_ = gradStep(gr, getGradient(dialect, info, env))
#     return dialect.putLoss(l, env)


@dataclass(frozen=True)
class RnnLibrary(Generic[ENV, DATA]):
    activationStep: Callable[[DATA, ENV], ENV]
    predictionStep: Callable[[DATA, ENV], ENV]
    lossStep: Callable[[DATA, ENV], ENV]

class _RnnDialect(HasActivation[ENV, ACTIV]
                , HasParameter[ENV, PARAM]
                , HasRecurrentWeights[ENV, PARAM, PARAM_R]
                , HasReadoutWeights[ENV, PARAM, PARAM_O]
                , HasPrediction[ENV, PRED]
                , HasLoss[ENV, LOSS]
                , Protocol[ENV, ACTIV, PRED, PARAM, PARAM_R, PARAM_O]):
    pass

class _RnnDataDialect(HasInput[DATA_L, X_L], HasPredictionInput[DATA_L, Y_L], HasLabel[DATA_L, Z_L], Protocol[DATA_L, X_L, Y_L, Z_L]):
    pass

# Smart constructor
def createRnnLibrary( dialect: _RnnDialect[ENV, ACTIV, PRED, PARAM, PARAM_R, PARAM_O]
                    , dataDialect: _RnnDataDialect[DATA, X, Y, Z]
                    , activationStep: Callable[[X, ACTIV, PARAM_R], ACTIV]
                    , predictionStep: Callable[[Y, ACTIV, PARAM_O], PRED]
                    , computeLoss: Callable[[Z, PRED], LOSS]) -> RnnLibrary[ENV, DATA]:
    actv_: Callable[[DATA, ENV], ENV] = lambda x, env: activation(dialect, dataDialect, activationStep, x, env)
    pred_: Callable[[DATA, ENV], ENV] = lambda x, env: prediction(dialect, dataDialect, predictionStep, x, env)
    loss_: Callable[[DATA, ENV], ENV] = lambda x, env: loss(dialect, dataDialect, computeLoss, add, x, env)
    return RnnLibrary(
        activationStep=actv_,
        predictionStep=pred_,
        lossStep=loss_)


# class _Gradient(HasParameter[ENV, PARAM], HasGradient[ENV, GRADIENT], Protocol[ENV, PARAM]):
#     pass

# def getGradient(dialect: _Gradient[ENV, PARAM],
#                 lossFn: Callable[[DATA, ENV], LOSS],
#                 info: DATA,
#                 env: ENV) -> ENV:
#     parameretrize = reparametrizeInput(lossFn, dialect.putParameter)
#     p = dialect.getParameter(env)
#     parameretrized: Callable[[DATA, PARAM], LOSS] = lambda x, param: parameretrize(x, env, param)
#     gr = torch.func.jacrev(parameretrized, argnums=1)(info, p)
#     return dialect.putGradient(gr, env)

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
# 2. should I average losses?
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