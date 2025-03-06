#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=wandb_sync
#SBATCH --cpus-per-task=4
#SBATCH --output=wandb_sync_%j.log   
#SBATCH --error=wandb_sync_%j.err    

# Navigate to the working directory
cd /scratch/wlp9800/wandb

# Sync all wandb runs
wandb sync --sync-all
