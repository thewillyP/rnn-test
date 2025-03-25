import wandb
import pandas as pd
import numpy as np
import json

# Define project and sweep details
entity = "wlp9800-new-york-university"
project = "rnn-test"
sweep_id = "j9gac6ff"

# Initialize wandb API
api = wandb.Api()

# Fetch sweep details directly
sweep = api.sweep(f"{entity}/{project}/{sweep_id}")

# Extract hessian_eigenvalues artifacts
hessian_matrices = {}

for run in sweep.runs:
    # Dynamically construct artifact identifier based on the run's name
    artifact_id = f"{entity}/{project}/run-{run.id}-hessian_eigenvalues:v0"
    artifact = api.artifact(artifact_id)
    artifact_dir = artifact.download()

    # Load the table as a Pandas DataFrame
    table_path = f"{artifact_dir}/hessian_eigenvalues.table.json"

    with open(table_path) as file:
        json_dict = json.load(file)

    df = pd.DataFrame(json_dict["data"], columns=json_dict["columns"])
    matrix = df.to_numpy()
    matrix = np.delete(matrix, 0, axis=0)  # axis=0 specifies row deletion

    hessian_matrices[run.id] = matrix

# Print the last 10 slices of the matrices
for run_id, matrix in hessian_matrices.items():
    # Assuming you want the last 10 rows
    print(f"Run {run_id} Hessian Eigenvalues Matrix Shape: {matrix.shape}")
    last_10_slices = matrix[0, :]  # Get the last 10 rows (slices) of the matrix
    print(last_10_slices)
    print("\n" + "-" * 50 + "\n")  # Separator between matrices
