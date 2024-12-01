#%%
from data import *
from rnn import *
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
    num_epochs: int
    learning_rate: float
    rnnConfig: RnnConfig
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]



def train(config: Config):
    ts = torch.arange(0, config.seq)
    dataGenerator = getRandomTask(config.task)

    train_loader = getDataLoaderIO(dataGenerator, config.t1, config.t2, ts, config.numTr, config.batch_size_tr)
    test_loader = getDataLoaderIO(dataGenerator, config.t1, config.t2, ts, config.numTe, 1)

    model = RNN(config.rnnConfig)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)  

    n_total_steps = len(train_loader)
    for epoch in range(config.num_epochs):
        for i, (x, y) in enumerate(train_loader):    
            outputs = model(x)
            loss = config.criterion(outputs, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 1 == 0:
                print (f'Epoch [{epoch+1}/{config.num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')



rc = RnnConfig(
    n_in=2,
    n_h=128,
    n_out=1,
    num_layers=2
)

config = Config(
    task=Random(),
    seq=10,
    numTr=100, 
    numVl=10, 
    numTe=10, 
    batch_size_tr=100, 
    batch_size_vl=2, 
    t1=1, 
    t2=2,
    num_epochs=500,
    learning_rate=0.001,
    rnnConfig=rc,
    criterion=torch.functional.F.mse_loss
)

train(config)

