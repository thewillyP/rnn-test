#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --job-name=oho_experiments
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --array=1
#SBATCH --output="/scratch/wlp9800/logs/%x-%A-%a.out"
#SBATCH --error="/scratch/wlp9800/logs/%x-%A-%a.err"

set -e

if [ -z "$USER" ]; then
    echo "Error: USER environment variable must be set!"
    exit 1
fi

if [ -z "$IMAGE" ]; then
    echo "Error: IMAGE environment variable must be set!"
    exit 1
fi

# Ensure WANDB_SWEEP_ID is passed as argument for wandb agent
if [ -z "$WANDB_SWEEP_ID" ]; then
    echo "Error: WANDB_SWEEP_ID environment variable must be set!"
    exit 1
fi

# Ensure WANDB_API_KEY is set
if [ -z "$WANDB_API_KEY" ]; then
    echo "Error: WANDB_API_KEY environment variable must be set!"
    exit 1
fi

# delay so that multiple jobs don't start at the same time, for rate limiting
sleep $(shuf -i 0-10 -n 1)

singularity run --nv --containall --cleanenv --writable-tmpfs \
  --env WANDB_API_KEY=${WANDB_API_KEY} \
  --env WANDB_DIR=/wandb_data \
  --bind /scratch/${USER}/wandb:/wandb_data \
  /scratch/${USER}/images/${IMAGE}.sif ${WANDB_SWEEP_ID}
