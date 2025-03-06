#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=wandb_sync
#SBATCH --cpus-per-task=4
#SBATCH --output=wandb_sync_%j.log   
#SBATCH --error=wandb_sync_%j.err    
#SBATCH --mail-type=END
#SBATCH --mail-user=wlp9800@nyu.edu

# Changes often
USER=wlp9800
IMAGE=devenv

singularity exec --containall --no-home --cleanenv \
--bind /scratch/wlp9800/wandb/wandb:/wandb \
/scratch/${USER}/images/${IMAGE}.sif bash -c \
"cd /wandb && wandb sync --sync-all"

