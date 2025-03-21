import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define the wandb directory path
wandb_dir = Path("/wandb_data")


def sync_run(run_path):
    """Function to sync a single run using wandb sync."""
    try:
        result = subprocess.run(
            ["wandb", "sync", run_path],
            cwd=wandb_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"Successfully synced {run_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error syncing {run_path}: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False


def list_and_sync_unsynced_runs():
    # Check if the wandb directory exists
    if not wandb_dir.exists():
        print(f"Error: Directory '{wandb_dir}' does not exist.")
        return

    try:
        # Run the wandb sync --show command with a large number to show all unsynced runs
        result = subprocess.run(
            ["wandb", "sync", "--show", "100000"],
            cwd=wandb_dir,  # Run the command in the /scratch/wandb directory
            capture_output=True,
            text=True,
            check=True,
        )

        # Use stderr directly as per your request
        output = result.stderr.strip().splitlines()

        # Parse the total number of unsynced runs
        total_unsynced = 0
        unsynced_runs = []

        for line in output:
            line = line.strip()
            if "Number of runs to be synced:" in line:
                try:
                    total_unsynced = int(line.split(":")[-1].strip())
                except ValueError:
                    print(f"Could not parse total from: {line}")
            # Check if the line contains "wandb/offline-run-"
            elif "wandb/offline-run-" in line:
                # Extract just the run path by removing the "wandb: " prefix
                run_path = line.replace("wandb: ", "").strip()
                unsynced_runs.append(run_path)

        # Print results
        if total_unsynced == 0:
            print(f"No unsynced runs reported by wandb in '{wandb_dir}'.")
            return
        elif not unsynced_runs:
            print(f"Found {total_unsynced} unsynced runs, but none listed in output.")
            return
        else:
            print(f"Unsynced runs found in '{wandb_dir}':")
            print(f"\nTotal unsynced runs parsed: {len(unsynced_runs)} (Reported by wandb: {total_unsynced})")
            if len(unsynced_runs) != total_unsynced:
                print(
                    f"WARNING: Parsed {len(unsynced_runs)} runs, but wandb reported {total_unsynced}. Output may be truncated or parsing failed."
                )

            # Parallel syncing with 20 threads
            print(f"\nStarting parallel sync with 20 threads for {len(unsynced_runs)} runs...")
            with ThreadPoolExecutor(max_workers=15) as executor:
                # Submit all sync tasks
                future_to_run = {executor.submit(sync_run, run): run for run in unsynced_runs}
                successful_syncs = 0

                # Process results as they complete
                for future in as_completed(future_to_run):
                    run = future_to_run[future]
                    try:
                        if future.result():
                            successful_syncs += 1
                    except Exception as e:
                        print(f"Unexpected error syncing {run}: {e}")

            print(f"\nSyncing complete: {successful_syncs}/{len(unsynced_runs)} runs successfully synced.")

    except subprocess.CalledProcessError as e:
        print(f"Error executing wandb sync --show: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    list_and_sync_unsynced_runs()
