#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=14G
#SBATCH --time=00:15:00
#SBATCH --job-name=build
#SBATCH --error=_build.err
#SBATCH --output=_build.log
#SBATCH --cpus-per-task=10
#SBATCH --mail-type=END
#SBATCH --mail-user=wlp9800@nyu.edu


IMAGE=rnn-test
DOCKER_URL="docker://thewillyp/${IMAGE}:master-1.0.41"

# Build the Singularity image
singularity build --force /scratch/${USER}/images/${IMAGE}-cpu.sif ${DOCKER_URL}-cpu
singularity build --force /scratch/${USER}/images/${IMAGE}-gpu.sif ${DOCKER_URL}-gpu

# Create the overlay
singularity overlay create --size 5120 /scratch/${USER}/images/${IMAGE}-cpu.sif
singularity overlay create --size 5120 /scratch/${USER}/images/${IMAGE}-gpu.sif


# 5 GiB