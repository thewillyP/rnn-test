#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --job-name=oho_experiments
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --array=1
#SBATCH --output="/scratch/wlp9800/logs/%x-%A-%a.out"
#SBATCH --error="/scratch/wlp9800/logs/%x-%A-%a.err"

set -e

if [ -z "$WANDB_SWEEP_ID" ]; then
    echo "Error: WANDB_SWEEP_ID environment variable must be set!"
    exit 1
fi

if [ -z "$VARIANT" ]; then
    echo "Error: VARIANT environment variable must be set!"
    exit 1
fi

source ~/.secrets/env.sh

singularity run --nv --containall --cleanenv --writable-tmpfs \
  --env WANDB_API_KEY=${WANDB_API_KEY} \
  --bind /scratch/${USER}/wandb:/wandb_data \
  /scratch/${USER}/images/rnn-test-${VARIANT}.sif ${WANDB_SWEEP_ID} http://${SWEEP_HOST}:${SWEEP_PORT}
