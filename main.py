#%%
from data import *
import torch

torch.manual_seed(0)
        
@dataclass
class Config:
    task: DatasetType
    seq: int
    numTr: int
    numVl: int
    numTe: int
    batch_size_tr: int
    batch_size_vl: int
    t1: float
    t2: float



def main(config: Config):
    ts = torch.arange(0, config.seq)
    dataGenerator = getRandomTask(config.task)

    train_loader = getDataLoaderIO(dataGenerator, config.t1, config.t2, ts, config.numTr, config.batch_size_tr)
    test_loader = getDataLoaderIO(dataGenerator, config.t1, config.t2, ts, config.numTe, 1)

    for i, (xs, ys) in enumerate(train_loader):
        visualizeOutput(ys)

config = Config(task=Random(), seq=10, numTr=100, numVl=10, numTe=10, batch_size_tr=100, batch_size_vl=2, t1=1, t2=2)

main(config)

# %%
