import wandb
import torch
import hashlib

# Initialize wandb
wandb.login()

# Function to download model artifact and load state_dict
def load_model_state(artifact_name, version):
    api = wandb.Api()
    artifact = api.artifact(f"wlp9800-new-york-university/rnn-test-again/{artifact_name}:{version}")
    artifact_dir = artifact.download()
    model_path = f"{artifact_dir}/model_init.pt"  # Adjust path if necessary
    state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    return state_dict

# Function to download dataset artifact and load the TensorDataset
def load_dataset(artifact_name, version):
    api = wandb.Api()
    artifact = api.artifact(f"wlp9800-new-york-university/rnn-test-again/{artifact_name}:{version}")
    artifact_dir = artifact.download()
    dataset_path = f"{artifact_dir}/dataset_train.pt"  # Adjust path if necessary
    dataset = torch.load(dataset_path, map_location=torch.device('cpu'))
    return dataset

# Function to compute a hash for a state_dict
def compute_state_hash(state_dict):
    state_bytes = torch.cat([p.flatten() for p in state_dict.values()]).numpy().tobytes()
    return hashlib.md5(state_bytes).hexdigest()

# Function to compute a hash for a TensorDataset
def compute_tensordataset_hash(dataset):
    if isinstance(dataset, torch.utils.data.TensorDataset):
        tensor_hashes = []
        for tensor in dataset.tensors:
            tensor_bytes = tensor.flatten().numpy().tobytes()
            tensor_hashes.append(hashlib.md5(tensor_bytes).hexdigest())
        combined_hash = hashlib.md5("".join(tensor_hashes).encode()).hexdigest()
        return combined_hash
    else:
        raise ValueError("Dataset is not a TensorDataset.")

# Load the models
state_dict_1 = load_model_state("model_ripybcfo", "v0")
state_dict_2 = load_model_state("model_hof1cg6g", "v0")

# Load the datasets
dataset_1 = load_dataset("dataset_ripybcfo", "v0")
dataset_2 = load_dataset("dataset_hof1cg6g", "v0")

# Compute hashes for models
hash_1 = compute_state_hash(state_dict_1)
hash_2 = compute_state_hash(state_dict_2)

# Compute hashes for datasets
dataset_hash_1 = compute_tensordataset_hash(dataset_1)
dataset_hash_2 = compute_tensordataset_hash(dataset_2)

# Compare model hashes
if hash_1 == hash_2:
    print("The models are identical.")
else:
    print("The models are different.")

# Compare dataset hashes
if dataset_hash_1 == dataset_hash_2:
    print("The datasets are identical.")
else:
    print("The datasets are different.")
