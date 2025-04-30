#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=00:25:00
#SBATCH --job-name=oho_experiments
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --array=1-2000
#SBATCH --output="/vast/wlp9800/logs/%x-%A-%a.out"
#SBATCH --error="/vast/wlp9800/logs/%x-%A-%a.err"

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
  --bind /scratch/${USER}/space:/dump \
  /scratch/${USER}/images/rnn-test-${VARIANT}.sif ${WANDB_SWEEP_ID} http://${SWEEP_HOST}:${SWEEP_PORT}
