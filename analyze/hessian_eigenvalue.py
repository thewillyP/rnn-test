import wandb
import numpy as np


def fetch_hessian_eigenvalues(sweep_ids):
    api = wandb.Api()
    results = {}

    for sweep_id in sweep_ids:
        sweep = api.sweep("wlp9800-new-york-university/rnn-test/" + sweep_id)
        results[sweep_id] = {}

        for run in sweep.runs:
            hessian_data = run.history(keys=["hessian_eigenvalues"], pandas=False)
            eigenvalues_per_epoch = []

            for entry in hessian_data:
                if "hessian_eigenvalues" in entry:
                    hist_values = entry["hessian_eigenvalues"]
                    print(hist_values)
                    quit()
                    if isinstance(hist_values, dict) and "values" in hist_values:
                        print(hist_values)
                        quit()
                        eigenvalues_per_epoch.append(np.array(hist_values["values"]))

            if eigenvalues_per_epoch:
                results[sweep_id][run.id] = {
                    "name": run.name,
                    "url": run.url,
                    "eigenvalues": np.stack(eigenvalues_per_epoch),
                }

    return results


def detect_explosion(eigenvalues_per_epoch, threshold=1e6):
    """Detects if the consecutive product of eigenvalues along any axis explodes."""
    # Compute the cumulative product of the eigenvalues
    cumprod_vals = np.cumprod(eigenvalues_per_epoch, axis=0)

    # Find the first epoch where any column exceeds the threshold
    exceeded_epochs = np.where(np.any(cumprod_vals > threshold, axis=1))[0]

    if len(exceeded_epochs) > 0:
        # Find the first epoch where the explosion happens
        t = exceeded_epochs[0]

        # Filter out the eigenvalues at the columns that caused the explosion
        exceeded_columns = np.where(cumprod_vals[t] > threshold)[0]

        # Return the filtered history up to the explosion (always as a NumPy array)
        return eigenvalues_per_epoch[: t + 1, exceeded_columns]

    # If no explosion happens, return an empty NumPy array
    return np.array([])  # This ensures filtered_eigenvalues is always a NumPy array


def print_eigenvalue_matrix(filtered_eigenvalues):
    """Prints the filtered eigenvalue matrix showing only the contributing eigenvalues."""
    print(f"\n--- Eigenvalue Matrix Leading to Explosion ---")
    print(f"Filtered Eigenvalue Matrix from t=0 to explosion epoch:")
    print(filtered_eigenvalues)


def main():
    sweep_ids = ["uxwbaksr", "i5pudld3", "dcyxi0lw", "cso36rsc"]
    data = fetch_hessian_eigenvalues(sweep_ids)

    for sweep_id, runs in data.items():
        for run_id, run_info in runs.items():
            eigenvalues_per_epoch = run_info["eigenvalues"]
            print(eigenvalues_per_epoch[1])
            quit()
            filtered_eigenvalues = detect_explosion(eigenvalues_per_epoch)

            # Check if filtered_eigenvalues is non-empty
            if filtered_eigenvalues.shape[0] > 0:
                print_eigenvalue_matrix(filtered_eigenvalues)
                print(f"Explosion detected in sweep {sweep_id}, run {run_info['name']} ({run_id})")
                print(f"Run link: {run_info['url']}")
            else:
                print(f"No explosion detected in sweep {sweep_id}, run {run_info['name']} ({run_id})")
                print(f"Run link: {run_info['url']}")


if __name__ == "__main__":
    main()
