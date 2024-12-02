#%%
from data import *
from rnn import *
import torch

torch.manual_seed(23423)
        
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
    optimizerFn: Callable



def train(config: Config, model: RNN):
    ts = torch.arange(0, config.seq)
    dataGenerator = getRandomTask(config.task)

    train_loader = getDataLoaderIO(dataGenerator, config.t1, config.t2, ts, config.numTr, config.batch_size_tr)
    test_loader = getDataLoaderIO(dataGenerator, config.t1, config.t2, ts, config.numTe, config.numTe)

    optimizer = config.optimizerFn(model.parameters(), lr=config.learning_rate)

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
    
    return model


def visualize(config: Config, model: RNN):
    ts = torch.arange(0, config.seq)
    dataGenerator = getRandomTask(config.task)
    test_loader = getDataLoaderIO(dataGenerator, config.t1, config.t2, ts, config.numTe, 1)
    xs, ys = next(iter(test_loader))
    predicts = model(xs)

    ys = ys[0]
    predicts = predicts[0]


    plt.plot(ts.detach().numpy(), ys.flatten().detach().numpy(), label='True Values')
    plt.plot(ts.detach().numpy(), predicts.flatten().detach().numpy(), label='Predictions')

    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.title('Time Series Prediction vs True Values')
    plt.legend()
    plt.show()


def test_loss(loader: DataLoader, config: Config, model: RNN):
    with torch.no_grad():  
        def getLoss(pair):
            inputs, targets = pair
            predictions = model(inputs) 
            return config.criterion(predictions, targets) * inputs.size(0)
        total = torch.vmap(getLoss)(loader)
        return total.sum().item() / len(loader.dataset)



rc = RnnConfig(
    n_in=2,
    n_h=30,
    n_out=1,
    num_layers=1
)

config = Config(
    task=Random(),
    seq=15,
    numTr=10000, 
    numVl=10, 
    numTe=1000, 
    batch_size_tr=10000, 
    batch_size_vl=2, 
    t1=3, 
    t2=5,
    num_epochs=500,
    learning_rate=0.001,
    rnnConfig=rc,
    criterion=torch.functional.F.mse_loss,
    optimizerFn=torch.optim.SGD
)

model = RNN(config.rnnConfig)
model = train(config, model)

#%%
visualize(config, model)
# %%
