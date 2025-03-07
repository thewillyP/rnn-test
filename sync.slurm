#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=12G
#SBATCH --time=05:00:00
#SBATCH --job-name=wandb_sync
#SBATCH --cpus-per-task=6
#SBATCH --output="/scratch/wlp9800/logs/wandb_sync_%j.out"   
#SBATCH --error="/scratch/wlp9800/logs/wandb_sync_%j.err"    
#SBATCH --mail-type=END
#SBATCH --mail-user=wlp9800@nyu.edu

# Changes often
USER=wlp9800
IMAGE=devenv

singularity exec --containall --no-home --cleanenv \
    --bind /scratch/${USER}:/scratch \
    --bind /home/${USER}/dev:/dev \
    --env WANDB_API_KEY=${WANDB_API_KEY} \
    /scratch/${USER}/images/${IMAGE}.sif bash -c \
    "cd /dev && python wandb_syncp.py"

