import wandb
import pandas as pd
import numpy as np
import json


def fetch_hessian_eigenvalues(sweep_ids):
    api = wandb.Api()
    results = {}

    for sweep_id in sweep_ids:
        sweep = api.sweep(f"wlp9800-new-york-university/rnn-test/{sweep_id}")
        results[sweep_id] = {}

        for run in sweep.runs:
            artifact_id = f"wlp9800-new-york-university/rnn-test/run-{run.id}-hessian_eigenvalues:v0"
            artifact = api.artifact(artifact_id)
            artifact_dir = artifact.download()

            table_path = f"{artifact_dir}/hessian_eigenvalues.table.json"
            with open(table_path) as file:
                json_dict = json.load(file)

            df = pd.DataFrame(json_dict["data"], columns=json_dict["columns"])
            matrix = df.to_numpy()
            matrix = np.delete(matrix, 0, axis=0)  # Remove the first row

            results[sweep_id][run.id] = {
                "name": run.name,
                "url": run.url,
                "eigenvalues": matrix,
            }
    return results


def detect_explosion(eigenvalues_per_epoch, threshold=1e6):
    cumprod_vals = np.cumprod(eigenvalues_per_epoch, axis=0)
    exceeded_epochs = np.where(np.any(cumprod_vals > threshold, axis=1))[0]

    if len(exceeded_epochs) > 0:
        t = exceeded_epochs[0]
        exceeded_columns = np.where(cumprod_vals[t] > threshold)[0]
        return eigenvalues_per_epoch[: t + 1, exceeded_columns]

    return np.array([])


def print_eigenvalue_matrix(filtered_eigenvalues):
    print("\n--- Eigenvalue Matrix Leading to Explosion ---")
    print("Filtered Eigenvalue Matrix from t=0 to explosion epoch:")
    print(filtered_eigenvalues)


def main():
    sweep_ids = ["ll2rh1tb", "24wnnntk", "z7twlfda", "j9gac6ff"]
    data = fetch_hessian_eigenvalues(sweep_ids)

    for sweep_id, runs in data.items():
        for run_id, run_info in runs.items():
            eigenvalues_per_epoch = run_info["eigenvalues"]
            filtered_eigenvalues = detect_explosion(eigenvalues_per_epoch)

            if filtered_eigenvalues.shape[0] > 0:
                # print_eigenvalue_matrix(filtered_eigenvalues)
                print(f"Explosion detected in sweep {sweep_id}, run {run_info['name']} ({run_id})")
                print(f"Run link: {run_info['url']}")
            else:
                print(f"No explosion detected in sweep {sweep_id}, run {run_info['name']} ({run_id})")
                print(f"Run link: {run_info['url']}")


if __name__ == "__main__":
    main()
