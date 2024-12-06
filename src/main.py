from typing import Any
from data import *
from learning import SGD, efficientBPTT_Vanilla_Full
from objectalgebra import OhoStateInterpreter
from rnn import *
import torch
import wandb
import argparse
from abc import ABC, abstractmethod
import os
from toolz import take
from records import ArtifactConfig, Logger, ZeroInit, RandomInit, RnnConfig, Config, Random, Sparse, Wave


        
class WandbLogger(Logger):
    
    def log2External(self, artifactConfig: ArtifactConfig, saveFile: Callable[[str], None], name: str):
        path = artifactConfig.path(name)
        saveFile(path)
        artifact = artifactConfig.artifact(wandb.run.id)
        artifact.add_file(path)
        wandb.log_artifact(artifact)
    
    def log(self, dict: dict[str, Any]):
        wandb.log(dict)
    
    def init(self, projectName: str, config: argparse.Namespace):
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(project=projectName, config=config)
    
    def watchPytorch(self, model: nn.Module):
        wandb.watch(model, log_freq=1000, log="all")

class PrettyPrintLogger(Logger):
        
    def log2External(self, artifactConfig: ArtifactConfig, saveFile: Callable[[str], None], name: str):
        print(f"Saving {name} to {artifactConfig.path(name)}")
        saveFile(artifactConfig.path(name))
        print(f"Saved {name} to {artifactConfig.path(name)}")

    def log(self, dict: dict[str, Any]):
        for key, value in dict.items():
            print(f"{key}: {value}")
    
    def init(self, projectName: str, config: argparse.Namespace):
        print(f"Initialized project {projectName} with config {config}")
    
    def watchPytorch(self, model: nn.Module):
        print("Model is being watched")


"""
xs: Iterator[Iterator[DATA@efficientBPTT_Vanilla]],
    env: ENV@efficientBPTT_Vanilla
) -> ENV@efficientBPTT_Vanilla

Iteratr1 = single dimension, time series split up size
Iterator2 = actual time series

[1, T, D]
just wrap dataset in a []
to add batch
[B, 1, T, D]
I just vmap
"""

def train(config: Config, logger: Logger, model: RNN):
    ts = torch.arange(0, config.seq)
    dataGenerator = getRandomTask(config.task)

    train_ds = getDatasetIO(dataGenerator, config.t1, config.t2, ts, config.numTr)
    train_loader = getDataLoaderIO(train_ds, config.batch_size_tr)
    test_ds = getDatasetIO(dataGenerator, config.t1, config.t2, ts, config.numTe)
    test_loader = getDataLoaderIO(test_ds, config.batch_size_te)

    log_datasetIO(config, logger, train_ds, "train")
    log_datasetIO(config, logger, test_ds, "test")

    # sgd = SGD(config.learning_rate)
    # bptt = efficientBPTT_Vanilla_Full(sgd, config.criterion, OhoStateInterpreter())

    optimizer = config.optimizerFn(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.num_epochs):
        for i, (x, y) in enumerate(train_loader):   

            # def closure(inputs, targets):

            outputs = model(x)
            loss = config.criterion(outputs, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch * len(train_loader) + i) % config.logFrequency == 0:
                logger.log({"loss": loss.item()
                        , "gradient_norm": gradient_norm(model)})
        
        logger.log({"test_loss": test_loss(config, test_loader, model)})
        if (epoch+1) % config.checkpointFrequency == 0:
            log_modelIO(config, logger, model, f"epoch_{epoch}")
            logger.log({"performance": visualize(config, model, test_ds)})

    
    return model


def visualize(config: Config, model: RNN, dataset: TensorDataset):
    ts = torch.arange(0, config.seq)
    samples = min(config.performanceSamples, len(dataset))
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    n_cols = 3
    n_rows = (samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() 
    
    for i, (xs, ys) in enumerate(take(samples, test_loader)):
        predicts = model(xs)
        ys = ys[0]
        predicts = predicts[0]
        
        ax = axes[i]
        ax.plot(ts.detach().numpy(), ys.flatten().detach().numpy(), label='True Values')
        ax.plot(ts.detach().numpy(), predicts.flatten().detach().numpy(), label='Predictions')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Values')
        ax.set_title(f"Predictions vs True Values for {config.task} (Sample {i+1})")
        ax.legend()
    # Hide any empty subplots if there are fewer samples than subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    fig.tight_layout()  # Adjust layout for better spacing
    img = wandb.Image(fig)
    plt.close(fig)
    return img

def test_loss(config: Config, loader: DataLoader, model: RNN):
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in loader:
            predictions = model(inputs)
            batch_loss = config.criterion(predictions, targets) * inputs.size(0)
            total_loss += batch_loss.item()
            total_samples += inputs.size(0)
    
    return total_loss / total_samples

def gradient_norm(model: RNN):
    grads = [
        param.grad.detach().flatten()
        for param in model.parameters()
        if param.grad is not None
    ]
    return torch.linalg.norm(torch.cat(grads), 2).item()


def parseIO():
    parser = argparse.ArgumentParser(description="Parse configuration parameters for RnnConfig and Config.")

    # Arguments for RnnConfig
    parser.add_argument('--n_in', type=int, required=True, help='Number of input features for the RNN')
    parser.add_argument('--n_h', type=int, required=True, help='Number of hidden units for the RNN')
    parser.add_argument('--n_out', type=int, required=True, help='Number of output features for the RNN')
    parser.add_argument('--num_layers', type=int, required=True, help='Number of layers in the RNN')

    # Arguments for Config
    parser.add_argument('--task', type=str, choices=['Random', 'Sparse', 'Wave'], required=True,
                        help="Task type (Random, Sparse, or Wave)")
    parser.add_argument('--init_scheme', type=str, choices=['ZeroInit', 'RandomInit'], required=True,
                        help="Rnn init scheme type (ZeroInit, or RandomInit)")
    parser.add_argument('--outT', type=int, help="Output time step (required if task is Sparse)")
    parser.add_argument('--seq', type=int, required=True, help='Sequence length')
    parser.add_argument('--numTr', type=int, required=True, help='Number of training samples')
    parser.add_argument('--numVl', type=int, required=True, help='Number of validation samples')
    parser.add_argument('--numTe', type=int, required=True, help='Number of testing samples')
    parser.add_argument('--batch_size_tr', type=int, required=True, help='Training batch size')
    parser.add_argument('--batch_size_vl', type=int, required=True, help='Validation batch size')
    parser.add_argument('--batch_size_te', type=int, required=True, help='Testing batch size')
    parser.add_argument('--t1', type=int, required=True, help='Parameter t1')
    parser.add_argument('--t2', type=int, required=True, help='Parameter t2')
    parser.add_argument('--num_epochs', type=int, required=True, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate')
    parser.add_argument('--optimizerFn', type=str, choices=['Adam', 'SGD'], required=True,
                        help="Optimizer function (Adam or SGD)")
    parser.add_argument('--lossFn', type=str, choices=['mse'], required=True,
                        help="Loss function (currently only mse is supported)")
    parser.add_argument('--mode', type=str, choices=['test', 'experiment'], required=True,
                        help="Execution mode (test or experiment)")
    parser.add_argument('--checkpoint_freq', type=int, required=True,
                        help="Frequency of checkpoints during training (in epochs)")
    parser.add_argument('--seed', type=int, required=True,
                        help="Pytorch seed")
    parser.add_argument('--projectName', type=str, required=True,
                        help="Wandb project name")
    parser.add_argument('--logger', type=str, choices=['wandb', 'prettyprint'], required=True,
                        help="Choice of logger to use")
    parser.add_argument('--performance_samples', type=int, required=True,
                        help="Number of samples to visualize performance")
    parser.add_argument('--activation_fn', type=str, choices=['relu', 'tanh'], required=True,
                        help="Activation function (relu or tanh)")
    parser.add_argument('--log_freq', type=int, required=True,
                        help="Frequency of logging during training (in iterations)")

    args = parser.parse_args()

    # Validate Sparse task requires outT
    if args.task == 'Sparse' and args.outT is None:
        parser.error("--outT is required when --task is Sparse")

    match args.lossFn:
        case 'mse':
            loss_function = torch.functional.F.mse_loss
        case _:
            raise ValueError("Currently only mse is supported as a loss function")

    # Determine optimizer function
    match args.optimizerFn:
        case 'Adam':
            optimizer_fn = torch.optim.Adam
        case 'SGD':
            optimizer_fn = torch.optim.SGD
        case _:
            raise ValueError("Currently only Adam and SGD are supported as optimizer functions")

    # Placeholder for task initialization
    match args.task:
        case 'Sparse':
            task = Sparse(args.outT)
        case 'Random':
            task = Random()
        case 'Wave':
            task = Wave()
        case _:
            raise ValueError("Invalid task type")
    
    match args.logger:
        case 'wandb':
            logger = WandbLogger()
        case 'prettyprint':
            logger = PrettyPrintLogger()
        case _:
            raise ValueError("Invalid logger type")
    
    match args.init_scheme:
        case 'ZeroInit':
            scheme = ZeroInit()
        case 'RandomInit':
            scheme = RandomInit()
        case _:
            raise ValueError("Invalid init type")
        
    match args.activation_fn:
        case 'relu':
            activation_fn = torch.relu
        case 'tanh':
            activation_fn = torch.tanh
        case _:
            raise ValueError("Invalid activation function")
    
    rnnConfig = RnnConfig(
        n_in=args.n_in,
        n_h=args.n_h,
        n_out=args.n_out,
        num_layers=args.num_layers,
        scheme=scheme,
        activation=activation_fn
    )

    config = Config(
        task=task,
        seq=args.seq,
        numTr=args.numTr,
        numVl=args.numVl,
        numTe=args.numTe,
        batch_size_tr=args.batch_size_tr,
        batch_size_vl=args.batch_size_vl,
        batch_size_te=args.batch_size_te,
        t1=args.t1,
        t2=args.t2,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        rnnConfig=rnnConfig,
        criterion=loss_function,
        optimizerFn=optimizer_fn,
        modelArtifact=ArtifactConfig(artifact=lambda name: wandb.Artifact(f"model_{name}", type="model"), path=lambda x: f"model_{x}.pt"),
        datasetArtifact=ArtifactConfig(artifact=lambda name: wandb.Artifact(f"dataset_{name}", type="dataset"), path=lambda x: f"dataset_{x}.pt"),
        checkpointFrequency=args.checkpoint_freq,
        projectName=args.projectName,
        seed=args.seed,
        performanceSamples=args.performance_samples,
        logFrequency=args.log_freq
    )

    return args, config, logger



def log_modelIO(config: Config, logger: Logger, model: RNN, name: str):
    logger.log2External(config.modelArtifact, lambda path: torch.save(model.state_dict(), path), name)

def log_datasetIO(config: Config, logger: Logger, dataset: TensorDataset, name: str):
    logger.log2External(config.datasetArtifact, lambda path: torch.save(dataset, path), name)


def main():
    args, config, logger = parseIO()
    torch.manual_seed(config.seed)

    logger.init(config.projectName, args)
    model = RNN(config.rnnConfig)  # IO, random 
    # logger.watchPytorch(model)
    log_modelIO(config, logger, model, "init")

    model = train(config, logger, model)

if __name__ == "__main__":
    main()