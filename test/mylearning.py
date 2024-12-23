from typing import Callable, TypeVar, Iterator, Protocol, Type
import torch
from myrecords import RnnConfig
from myobjectalgebra import HasActivation, HasParameter, HasReadoutWeights, HasTrainingInput, HasTrainingLabel, HasRecurrentWeights
from myfunc import foldr, scan0, fst, snd
import itertools
from mytypes import *



ACTIV = TypeVar('ACTIV')
PRED = TypeVar('PRED')
PARAM = TypeVar('PARAM')
PARAM_T = TypeVar('PARAM_T')

class _Activation(HasActivation[ENV, ACTIV], HasParameter[ENV, PARAM], HasRecurrentWeights[ENV, PARAM, PARAM_T], Protocol[ENV, ACTIV, PARAM, PARAM_T]):
    pass

def activation(dialect: _Activation[ENV, ACTIV, PARAM, PARAM_T],
            step: Callable[[DATA, ACTIV, PARAM_T], ACTIV],
            env: ENV,
            info: DATA) -> ACTIV:
    a = dialect.getActivation(env)
    w_rec = dialect.getRecurrentWeights(env, dialect.getParameter(env))
    return step(info, a, w_rec)


class _Prediction(HasActivation[ENV, ACTIV], HasParameter[ENV, PARAM], HasReadoutWeights[ENV, PARAM, PARAM_T], Protocol[ENV, ACTIV, PARAM, PARAM_T]):
    pass

def prediction(dialect: _Prediction[ENV, ACTIV, PARAM, PARAM_T],
                        step: Callable[[DATA, ACTIV, PARAM_T], PRED],
                        env: ENV,
                        info: DATA) -> PRED:
    a = dialect.getActivation(env)
    w_out = dialect.getReadoutWeights(env, dialect.getParameter(env))
    return step(info, a, w_out)


# getGradient ::
#   (HasParameter env param) =>
#   (env -> info -> Loss) ->
#   env ->
#   info ->
#   Gradient
# getGradient lossFn env0 xs =
#   let parameretrize env info param =
#         let env' = putParameter env param
#          in lossFn env' info
#       p = getParameter env0
#       tensorInputFn = parameretrize env0
#    in gradient tensorInputFn xs p


def getGradient( dialect: HasParameter[ENV, PARAM],
                lossFn: Callable[[ENV, DATA], LOSS],
                env: ENV,
                info: DATA) -> GRADIENT:
    def parameretrize(env: ENV, info: DATA, param: PARAM) -> LOSS:
        env_ = dialect.putParameter(param, env)
        return lossFn(env_, info)
    p = dialect.getParameter(env)
    parameretrized = lambda x, param: parameretrize(env, x, param)
    return torch.func.jacrev(parameretrized, argnums=1)(info, p)


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