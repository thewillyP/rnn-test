from copy import deepcopy
from itertools import cycle
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

def update_optimizer_hyperparams(model, optimizer):
    optimizer.param_groups[0]['lr'] = model.eta
    optimizer.param_groups[0]['weight_decay'] = model.lambda_l2
    return optimizer

def unflatten_array(X, N, param_shapes):
    """Takes flattened array and returns in natural shape for network
    parameters."""
    return [torch.reshape(X[N[i]:N[i + 1]], s) \
        for i, s in enumerate(param_shapes)]

def flatten_array_w_0bias(X):
    """Takes list of arrays in natural shape of the network parameters
    and returns as a flattened 1D numpy array."""
    vec = []
    for x in X:
        if len(x.shape) == 0 :
            vec.append(torch.zeros(x.shape).flatten())
        else:
            vec.append(x.flatten())
    return torch.cat(vec)


def compute_HessianVectorProd(config: Config, model, dFdS, data, target, name:str):

    eps_machine = torch.finfo(data.dtype).eps

    ## Compute Hessian Vector product h
    vmax_x, vmax_d = 0, 0

    model_plus = deepcopy(model)
    for param, direction in zip(model_plus.parameters(), dFdS):
        vmax_x = max(vmax_x, torch.max(torch.abs(param)).item())
        vmax_d = max(vmax_d, torch.max(abs(direction)).item())
        break
    wandb.log({f"{name}_vmax_x": vmax_x, f"{name}_vmax_d": vmax_d}, commit=False)

    if vmax_d ==0: vmax_d = 1
    Hess_est_r = (eps_machine ** 0.5) * (1+vmax_x) / vmax_d
    Hess_est_r = max([ Hess_est_r, 0.001])
    wandb.log({f"{name}_Hess_est_r": Hess_est_r}, commit=False)
    for ((parameter_name, param), direction) in zip(model_plus.named_parameters(), dFdS):
        perturbation =  Hess_est_r * direction
        wandb.log({f"{name}_perturbation_{parameter_name}": torch.linalg.norm(perturbation, 2).item()}, commit=False)
        wandb.log({f"{name}_peturbation_max_{parameter_name}": torch.max(perturbation).item()}, commit=False)
        wandb.log({f"{name}_peturbation_min_{parameter_name}": torch.min(perturbation).item()}, commit=False)
        param.data.add_(perturbation)

    model_plus.train()
    loss = trainStepIO(config, data, target, model_plus)
    wandb.log({f"{name}_model_plus_loss": loss}, commit=False)

    model_minus = deepcopy(model)
    for param, direction in zip(model_minus.parameters(), dFdS):
        perturbation =  Hess_est_r * direction
        param.data.add_(-perturbation)
    
    model_minus.train()
    loss = trainStepIO(config, data, target, model_minus)
    wandb.log({f"{name}_model_minus_loss": loss}, commit=False)

    g_plus  = get_grads(model_plus)
    g_minus = get_grads(model_minus)
    wandb.log({f"{name}_g_plus_norm": torch.linalg.norm(g_plus, 2).item()}, commit=False)
    wandb.log({f"{name}_g_minus_norm": torch.linalg.norm(g_minus, 2).item()}, commit=False)

    Hv = (g_plus - g_minus) / (2 * Hess_est_r)
    
    return Hv 


def get_grad_valid(config: Config, model, data, target):

    val_model = deepcopy(model)
    val_model.train()
    val_loss = trainStepIO(config, data, target, val_model)
    grad_val = get_grads(val_model)

    wandb.log({"validation_loss": val_loss}, commit=False)
    
    return grad_val

def meta_update(config: Config, data_vl, target_vl, data_tr, target_tr, model, optimizer):

    #Compute Hessian Vector Product
    param_shapes = model.param_shapes
    dFdlr = unflatten_array(model.dFdlr, model.param_cumsum, param_shapes)
    Hv_lr  = compute_HessianVectorProd(config, model, dFdlr, data_tr, target_tr, "lr")

    dFdl2 = unflatten_array(model.dFdl2, model.param_cumsum, param_shapes)
    Hv_l2  = compute_HessianVectorProd(config, model, dFdl2, data_tr, target_tr, "l2")

    grad_valid = get_grad_valid(config, model, data_vl, target_vl)

    grad = get_grads(model)
    param = torch.nn.utils.parameters_to_vector(model.parameters()).data

    #Update hyper-parameters   
    model.update_dFdlr(Hv_lr, param, grad)
    model.update_dFdlambda_l2(Hv_l2, param)

    if config.is_oho:
        model.update_eta(config.meta_learning_rate, grad_valid)
        model.update_lambda(config.meta_learning_rate, grad_valid)
        optimizer = update_optimizer_hyperparams(model, optimizer)

    return model, optimizer, grad_valid


def trainStepIO(config: Config, x, y, model):
    xs = torch.chunk(x, config.time_chunk_size, dim=1)
    ys = torch.chunk(y, config.time_chunk_size, dim=1)
    s = model.getInitialActivation(x.size(0))
    real_loss = 0
    for xi, yi in zip(xs, ys):
        outputs = model(xi, s)
        myloss = config.criterion(outputs, yi)
        loss = myloss / config.time_chunk_size
        real_loss += myloss.item()
        loss.backward()
        s = model.activations[:, -1, :].clone().detach().unsqueeze(0)
    return real_loss

def train(config: Config, logger: Logger, model: RNN, train_loader: Iterator, validation_loader: Iterator, test_loader: Iterator, test_ds: TensorDataset):
    optimizer = config.optimizerFn(model.parameters(), lr=config.learning_rate, weight_decay=config.l2_regularization)
    optimizer = update_optimizer_hyperparams(model, optimizer)

    dataGenerator_ = getRandomTask(Wave())
    ts_ = torch.arange(0, config.seq)
    test_ds_ = getDatasetIO(dataGenerator_, config.t1, config.t2, ts_, config.numTe)
    test_loader_ = getDataLoaderIO(test_ds_, config.batch_size_te)

    for epoch in range(config.num_epochs):
        the_test_loss = test_loss(config, test_loader, model)
        counter = 0
        for i, (x, y) in enumerate(train_loader):   
            optimizer.zero_grad()
            real_loss = 0
            counter += x.size(0)
            real_loss = trainStepIO(config, x, y, model)
            real_loss *= x.size(0)
            optimizer.step()

            data_vl, target_vl = next(validation_loader)
            try:
                model, optimizer, grad_valid = meta_update(config, data_vl, target_vl, x, y, model, optimizer)
            except:
                pass

            if (epoch * len(train_loader) + i) % config.logFrequency == 0:
                log_data = {"loss": real_loss / counter
                        , "gradient_norm": gradient_norm(model)
                        , "learning_rate": optimizer.param_groups[0]['lr']
                        , "l2_regularization": optimizer.param_groups[0]['weight_decay']
                        , "validation_gradient_norm": torch.linalg.norm(grad_valid, 2).item()
                        , "dFdlr_tensor_norm": torch.linalg.norm(model.dFdlr, 2).item()
                        , "dFdl2_tensor_norm": torch.linalg.norm(model.dFdl2, 2).item()
                        , "parameter_norm": torch.linalg.norm(torch.nn.utils.parameters_to_vector(model.parameters()), 2).item()
                        , "iteration": epoch * len(train_loader) + i
                        , "epoch": epoch}
                
                # l2 = lambda x: torch.linalg.norm(x, 2)
                # vmapped = torch.vmap(torch.vmap(l2))(model.activations)
                # activations = {"activation_norms": vmapped.mean().item(), "activation_max": vmapped.max().item(), "activation_min": vmapped.min().item()}
                # log_data.update(activations)

                logger.log(log_data)
        
        logger.log({"test_loss": the_test_loss,
                    "wave_test_loss": wave_test_loss(config, model, test_loader_),
                    "epoch": epoch})
        if (epoch+1) % config.checkpointFrequency == 0:
            log_modelIO(config, logger, model, f"epoch_{epoch}")
            logger.log({"performance": visualize(config, model, test_ds),
                        "wave_performance": visualize(config, model, test_ds_),
                        "epoch": epoch})

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
        predicts = model(xs, model.getInitialActivation(xs.size(0)))
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
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            predictions = model(inputs, model.getInitialActivation(inputs.size(0)))
            batch_loss = config.criterion(predictions, targets) * inputs.size(0)
            total_loss += batch_loss.item()
            total_samples += inputs.size(0)
    
    model.train()
    return total_loss / total_samples

def wave_test_loss(config: Config, model: RNN, test_loader: DataLoader):
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            predictions = model(inputs, model.getInitialActivation(inputs.size(0)))
            batch_loss = config.criterion(predictions, targets) * inputs.size(0)
            total_loss += batch_loss.item()
            total_samples += inputs.size(0)
    
    return total_loss / total_samples


def get_grads(model: torch.nn.Module):
    grads = [
        param.grad.detach().flatten()
        for param in model.parameters()
        if param.grad is not None
    ]
    return torch.cat(grads)

def gradient_norm(model: RNN):
    grads = get_grads(model)
    return torch.linalg.norm(grads, 2).item()


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
    parser.add_argument('--randomType', type=str, choices=['Uniform', 'Normal'], required=False,
                        help="Random type (Uniform or Normal)")
    parser.add_argument('--init_scheme', type=str, choices=['ZeroInit', 'RandomInit', 'StaticRandomInit'], required=True,
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
    parser.add_argument('--l2_regularization', type=float, required=True, help='Learning rate')
    parser.add_argument('--meta_learning_rate', type=float, required=True, help='Meta Learning rate')
    parser.add_argument('--is_oho', type=int, required=True, help='Is OHO')
    parser.add_argument('--time_chunk_size', type=int, required=True, help='Time chunk size')

    args = parser.parse_args()

    # Validate Sparse task requires outT
    if args.task == 'Sparse' and args.outT is None:
        parser.error("--outT is required when --task is Sparse")
    
    if args.task == 'Random' and args.randomType is None:
        parser.error("--randomType is required when --task is Random")
    
    match args.is_oho:
        case 0:
            is_oho = False
        case 1:
            is_oho = True
        case _:
            raise ValueError("Invalid is_oho value")

    match args.lossFn:
        case 'mse':
            loss_function = lambda x, y: torch.functional.F.mse_loss(x, y, reduction='none').sum(dim=1).mean(dim=0)
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
            match args.randomType:
                case 'Uniform':
                    randType = Uniform()
                case 'Normal':
                    randType = Normal()
                case _:
                    raise ValueError("Invalid random type")
            task = Random(randType)
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
        case 'StaticRandomInit':
            scheme = StaticRandomInit()
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
        logFrequency=args.log_freq,
        l2_regularization=args.l2_regularization,
        meta_learning_rate=args.meta_learning_rate,
        is_oho=is_oho,
        time_chunk_size=args.time_chunk_size
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
    model = RNN(config.rnnConfig, config.learning_rate, config.l2_regularization) # Random IO 
    # logger.watchPytorch(model)
    log_modelIO(config, logger, model, "init")


    ts = torch.arange(0, config.seq)
    dataGenerator = getRandomTask(config.task)

    train_ds = getDatasetIO(dataGenerator, config.t1, config.t2, ts, config.numTr)
    train_loader = getDataLoaderIO(train_ds, config.batch_size_tr)
    test_ds = getDatasetIO(dataGenerator, config.t1, config.t2, ts, config.numTe)
    test_loader = getDataLoaderIO(test_ds, config.batch_size_te)
    valid_ds = getDatasetIO(dataGenerator, config.t1, config.t2, ts, config.numVl)
    valid_loader = getDataLoaderIO(valid_ds, config.batch_size_vl)

    log_datasetIO(config, logger, train_ds, "train")
    log_datasetIO(config, logger, test_ds, "test")
    log_datasetIO(config, logger, valid_ds, "valid")


    model = train(config, logger, model, train_loader, cycle(valid_loader), test_loader, test_ds)



if __name__ == "__main__":
    main()