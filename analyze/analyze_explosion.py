import os
import wandb
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Constants
PROJECT = "wlp9800-new-york-university/rnn-test"
SWEEP_IDS = {
    "nh_32": "p3smgatf",
    "nh_64": "eosnfwrt",
    "nh_128": ["uf083n2r", "jpz90pv1"],
    "nh_256": "ofyehg6k",
}
CACHE_DIR = "wandb_cache"
WINDOW_SIZE = 20  # For psi window
LOSS_WINDOW_SIZE = 100  # For slope and loss window
PSI_PAD = 5  # Padding before instability for psi
LOSS_PAD = 25  # Padding before instability for loss
THRESHOLD = 500
TARGET_LR = 1e-3  # Filter for this outer_learning_rate

# Setup
wandb.login()
api = wandb.Api()
os.makedirs(CACHE_DIR, exist_ok=True)


# Cache helper
def get_cache_file(run_id):
    return os.path.join(CACHE_DIR, f"{run_id}_history.pkl")


# Load or fetch run data
def load_history(run):
    cache_file = get_cache_file(run.id)
    if os.path.exists(cache_file):
        print(f"Loading cached data for {run.id}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print(f"Fetching data for {run.id}")
    history = pd.DataFrame(run.history())
    if history.empty:
        print(f"No data for {run.id}, skipping")
        return None

    with open(cache_file, "wb") as f:
        pickle.dump(history, f)
    return history


# Get runs directly from sweeps with filter
runs = []
for network_size, sweep_info in SWEEP_IDS.items():
    sweep_ids = [sweep_info] if isinstance(sweep_info, str) else sweep_info
    for sweep_id in sweep_ids:
        sweep = api.sweep(f"{PROJECT}/{sweep_id}")
        filtered_runs = [run for run in sweep.runs if run.config.get("outer_learning_rate", None) == TARGET_LR]
        runs.extend(filtered_runs)
print(f"Got {len(runs)} runs from sweeps (filtered for outer_learning_rate={TARGET_LR})")


# Core analysis functions
def get_instability_epochs(history):
    return history[history["outer_influence_tensor_norm"] > THRESHOLD].index


def get_run_stats(history, epoch):
    # Adjust epoch with padding
    psi_end = max(epoch - PSI_PAD, 0)
    loss_end = max(epoch - LOSS_PAD, 0)

    # Stats for scatter plot (100 epochs for slope, up to loss_end)
    slope_start = max(loss_end - LOSS_WINDOW_SIZE, 0)
    psi_short = history["largest_influence_eigenvalue"].iloc[slope_start : loss_end + 1]
    losses = history["train_loss"].iloc[slope_start : loss_end + 1]
    avg_psi = np.mean(psi_short[-WINDOW_SIZE:]) if len(psi_short) >= WINDOW_SIZE else np.mean(psi_short)

    # Fit a line to the full loss series for slope
    x = np.arange(len(losses))
    if len(losses) > 1:
        slope, _ = np.polyfit(x, losses, 1)  # Slope from linear fit
    else:
        slope = 0

    # 20-epoch window for eigenvalue plot (up to psi_end)
    psi_start = max(psi_end - WINDOW_SIZE, 0)
    psi_window = history["largest_influence_eigenvalue"].iloc[psi_start : psi_end + 1]

    # 100-epoch window for loss plot (up to loss_end)
    loss_window = losses

    return avg_psi, slope, psi_window, loss_window  # psi_window (20), loss_window (100)


def is_run_unstable(history):
    return len(history[history["outer_influence_tensor_norm"] > THRESHOLD]) > 0


# Main analysis
data = {size: [] for size in SWEEP_IDS.keys()}
run_counts = {size: 0 for size in SWEEP_IDS.keys()}
unstable_counts = {size: 0 for size in SWEEP_IDS.keys()}

for run in runs:
    history = load_history(run)
    if history is None:
        continue

    try:
        n_h = run.config["n_h"]
        network_size = f"nh_{n_h}"
    except KeyError:
        print(f"No 'n_h' in config for {run.id}, skipping")
        continue

    if network_size not in SWEEP_IDS:
        print(f"Network size {network_size} not in SWEEP_IDS, skipping {run.id}")
        continue

    # Count total runs and unstable runs
    run_counts[network_size] += 1
    if is_run_unstable(history):
        unstable_counts[network_size] += 1

    instability_epochs = get_instability_epochs(history)
    # Only process runs with instability for plots
    if instability_epochs.any():
        first_instability = instability_epochs[0]  # Take first instability
        avg_psi, slope, psi_window, loss_window = get_run_stats(history, first_instability)
        if avg_psi and slope is not None:
            data[network_size].append((avg_psi, slope))

        # Plot eigenvalue window (20 epochs, padded by PSI_PAD)
        epochs = range(max(first_instability - WINDOW_SIZE - PSI_PAD, 0), max(first_instability - PSI_PAD, 0) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, psi_window, label=f"Instability at {first_instability}")
        plt.xlabel("Epoch")
        plt.ylabel("Largest Influence Eigenvalue")
        plt.title(f"Largest Eigenvalue Before Instability for {network_size} Run {run.id} (Pad={PSI_PAD})")
        plt.legend()
        plt.savefig(f"{network_size}_{run.id}_eigenvalue_window.png")
        plt.close()

        # Plot training loss window (100 epochs, padded by LOSS_PAD)
        loss_epochs = range(
            max(first_instability - LOSS_WINDOW_SIZE - LOSS_PAD, 0), max(first_instability - LOSS_PAD, 0) + 1
        )
        plt.figure(figsize=(10, 6))
        plt.plot(loss_epochs, loss_window, label=f"Instability at {first_instability}")
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title(f"Training Loss Before Instability for {network_size} Run {run.id} (100 Epochs, Pad={LOSS_PAD})")
        plt.legend()
        plt.savefig(f"{network_size}_{run.id}_loss_window.png")
        plt.close()

# Compute berserk fraction as unstable runs over total runs
berserk_fractions = {
    size: unstable_counts[size] / run_counts[size] if run_counts[size] > 0 else 0 for size in SWEEP_IDS.keys()
}

# Print berserk fractions
print("\nBerserk Fractions (Unstable Runs / Total Runs, outer_learning_rate=1e-3):")
for size, fraction in berserk_fractions.items():
    print(f"{size}: {fraction:.4f} ({unstable_counts[size]}/{run_counts[size]})")

# Plotting: Scatter plot of avg_psi vs slope, one point per run
for size, points in data.items():
    if not points:
        continue

    psi_vals, slopes = zip(*points)
    plt.figure(figsize=(10, 6))
    plt.scatter(psi_vals, slopes, s=100)  # One point per run
    plt.xlabel(f"Average Largest Influence Eigenvalue (Psi, {WINDOW_SIZE} epochs, Pad={PSI_PAD})")
    plt.ylabel(f"Slope of Training Curve Before Instability ({LOSS_WINDOW_SIZE} epochs, Pad={LOSS_PAD})")
    plt.title(f"Scatterplot for {size} Runs with Instability (outer_learning_rate=1e-3)")
    plt.savefig(f"{size}_scatterplot.png")
    plt.close()

print("Done, check your PNGs.")
