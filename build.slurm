#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=14G
#SBATCH --time=00:15:00
#SBATCH --job-name=build
#SBATCH --error=build.err
#SBATCH --output=build.log
#SBATCH --cpus-per-task=10
#SBATCH --mail-type=END
#SBATCH --mail-user=wlp9800@nyu.edu

# Dynamically pass user, image, and tag as environment variables
USER=${USER:-"wlp9800"}  # Default to wlp9800 if not provided
IMAGE=${IMAGE:-"rnn-test"}  # Default to rnn-test if not provided
TAG=${TAG:-"master-1.0.40-gpu"}  # Default to master-1.0.39-gpu if not provided

# Docker URL constructed dynamically with IMAGE and TAG
DOCKER_URL="docker://thewillyp/${IMAGE}:${TAG}"
SAVE_DIR="/scratch/${USER}/"

# Build the Singularity image
singularity build --force /scratch/${USER}/images/${IMAGE}.sif ${DOCKER_URL}

# Create the overlay
singularity overlay create --size 10240 /scratch/${USER}/images/${IMAGE}.sif

# 10 GiB