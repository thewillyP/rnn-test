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

# Load the models
state_dict_1 = load_model_state("model_ripybcfo", "v0")
state_dict_2 = load_model_state("model_hof1cg6g", "v0")

# state_dict_2 = load_model_state("model_j67ae882", "v0")



# Function to compute a hash of the state_dict
def compute_state_hash(state_dict):
    state_bytes = torch.cat([p.flatten() for p in state_dict.values()]).numpy().tobytes()
    return hashlib.md5(state_bytes).hexdigest()

# Compute hashes
hash_1 = compute_state_hash(state_dict_1)
hash_2 = compute_state_hash(state_dict_2)

# Compare hashes
if hash_1 == hash_2:
    print("The models are identical.")
else:
    print("The models are different.")
