from typing import Callable
from toolz import curry, compose
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import TypeVar, Callable, Generic, Generator, Iterator
from dataclasses import dataclass


@curry
def createUnitSignal(startTime: float, duration: float) -> Callable[[float], float]:
    def unit(t):
        return ((0 <= t - startTime) & (t - startTime <= duration)) 
    return unit

@curry 
def createSparseSignal(amplitude: float, t0: float, duration: float):
    return compose(lambda x: amplitude*x, createUnitSignal(t0, duration))


def sparseSignal( startOffset: float
                , a: float
                , dur: float
                , outT: float):
    return createSparseSignal(a, outT - startOffset, dur)


def sinWave(amplitude: float, frequency: float, phase_shift: float, bias: float):
    return lambda t: amplitude * torch.sin(frequency * t + phase_shift) + bias


@dataclass(frozen=True)
class WaveInitIO:
    initAmplitudeIO: Callable[[], float]
    initFrequencyIO: Callable[[], float]
    initPhaseShiftIO: Callable[[], float]
    initBiasIO: Callable[[], float]

@dataclass(frozen=True)
class SparseInitIO:
    initAmplitudeIO: Callable[[], float]
    initT0IO: Callable[[float], float]
    initDurationIO: Callable[[], float]
    outTIO : Callable[[], float]


def waveIO(waveInitIO: WaveInitIO):
    amplitude = waveInitIO.initAmplitudeIO()
    frequency = waveInitIO.initFrequencyIO()
    phase_shift = waveInitIO.initPhaseShiftIO()
    bias = waveInitIO.initBiasIO()
    return lambda t, _: sinWave(amplitude, frequency, phase_shift, bias)(t)


def sparseIO(sparseInitIO: SparseInitIO):
    outT = sparseInitIO.outTIO()
    def sparseFn(ts, t0):
        t0 = sparseInitIO.initT0IO(t0)
        amplitude = sparseInitIO.initAmplitudeIO()
        duration = sparseInitIO.initDurationIO()
        return sparseSignal(t0, amplitude, duration, outT)(ts)
    return sparseFn, sparseFn



def createDelayAddExample(t1, t2, ts, getRandFn):
    randFn1, randFn2 = getRandFn()
    x1 = randFn1(ts, t1)
    x2 = randFn2(ts, t2)
    y = torch.roll(x1, shifts=t1, dims=0) + torch.roll(x2, shifts=t2, dims=0)  # giving up on mathematically elegant y(t) = x(t - t1) + x(t - t2) bc computer can't support arbitrary precision. From now use integers only
    mask = ts >= max(t1, t2) 
    y = y * mask 
    return x1, x2, y


def createExamples(n, randomMonad):
    XS1, XS2, YS = torch.vmap(lambda _: randomMonad(), randomness='different')(torch.arange(n))
    combined_XS = torch.stack((XS1, XS2), dim=-1)
    return combined_XS, YS.unsqueeze(-1)



def visualizeOutput(YS):
    from matplotlib.ticker import MaxNLocator

    for ys in YS:  # Loop over each example
        plt.plot(range(ys.size(0)), ys, 'o-')


    plt.title('Output Time Series for All Examples')
    plt.xlabel('Time')
    plt.ylabel('Output Value')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()


sparseUniformConstOutT = lambda outT: SparseInitIO(lambda: torch.rand(1) - 0.5, lambda x: x, lambda: 1, lambda: outT)
sparseUniform = SparseInitIO(lambda: torch.rand(1) - 0.5, lambda x: x, lambda: 1, lambda: torch.randint(5, 10, (1,)))
waveArbitraryUniform = WaveInitIO(lambda: torch.rand(1), lambda: torch.rand(1)*100, lambda: torch.rand(1)*2*torch.pi, lambda: torch.rand(1)*2 - 1)
randomUniform = lambda ts, _: torch.rand(len(ts)) - 0.5
randomNormal = lambda ts, _: torch.randn(len(ts))
# later in the future I can create a config file instead of having to manually change this code everytime

# @curry
def getDatasetIO(randFn, t1: int, t2: int, ts: torch.Tensor, numEx: int):
    gen = lambda: createDelayAddExample(t1, t2, ts, randFn)
    XS, YS = createExamples(numEx, gen)
    ds = TensorDataset(XS, YS)
    return ds

def getDataLoaderIO(ds: TensorDataset, batchSize: int):
    return DataLoader(ds, batch_size=batchSize, shuffle=True, drop_last=True)


@dataclass
class Random:
    pass

@dataclass
class Sparse:
    outT: float

@dataclass
class Wave:
    pass

DatasetType = Random | Sparse | Wave

def getRandomTask(task: DatasetType):
    match task:
        case Random():
            return lambda: (randomUniform, randomUniform)
        case Sparse(outT):
            return lambda: sparseIO(sparseUniformConstOutT(outT))
        case Wave():
            return lambda: (waveIO(waveArbitraryUniform), waveIO(waveArbitraryUniform))
        case _:
            raise Exception("Invalid dataset type")