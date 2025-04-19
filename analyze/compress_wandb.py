import wandb
import os
import gzip
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Parameters
download_dir = "/scratch/downloaded_artifacts"
results_dir = "/scratch/results"
entity = "wlp9800-new-york-university"
project_name = "oho_exps"
group_name = "time_test_oho_d1efc05e0903463ca4e95a52714389a0"

# Ensure directories exist
os.makedirs(download_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)


def process_run(run):
    api = wandb.Api()
    run_path = f"{entity}/{project_name}/{run.id}"
    run_api = api.run(run_path)

    try:
        # Step 1: Download artifact logs_{run.id}:v0
        artifact_name = f"logs_{run.id}:v0"
        artifact = api.artifact(f"{run_path}/{artifact_name}")
        download_path = artifact.download(root=download_dir)
        log_file = os.path.join(download_path, "logs.pkl")

        if not os.path.exists(log_file):
            print(f"Log file not found for run {run.id}")
            return

        # Step 2: Compress the file with gzip
        compressed_file = os.path.join(results_dir, f"logs_{run.id}.pkl.gz")
        with open(log_file, "rb") as f_in:
            with gzip.open(compressed_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Step 3: Upload compressed file as new artifact
        new_artifact = wandb.Artifact(f"logs_gz_{run.id}", type="compressed_logs")
        new_artifact.add_file(compressed_file)
        with wandb.init(project=project_name, entity=entity, job_type="upload_compressed"):
            run_api.log_artifact(new_artifact)

        # Step 4: Delete original artifact
        for art in run_api.logged_artifacts():
            if art.name == f"logs_{run.id}:v0":
                art.delete(delete_aliases=True)
                break

        print(f"Successfully processed run {run.id}")

    except Exception as e:
        print(f"Error processing run {run.id}: {str(e)}")


def main():
    api = wandb.Api()

    # Get all runs in the project with the specified group
    runs = api.runs(f"{entity}/{project_name}", {"group": group_name})

    # Process runs in parallel
    with ThreadPoolExecutor() as executor:
        executor.map(process_run, runs)


if __name__ == "__main__":
    main()
