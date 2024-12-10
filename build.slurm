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

USER=wlp9800
IMAGE="rnn-test"
DOCKER_URL="docker://thewillyp/${IMAGE}:master-1.0.19@sha256:ccbd88a81bdedd44c4c440775e40528d86eb757af60e9ca42e6eef47fa83e396"
SAVE_DIR="/scratch/${USER}/"


singularity build --force /scratch/${USER}/images/${IMAGE}.sif ${DOCKER_URL} 
singularity overlay create --size 25600 /scratch/${USER}/images/${IMAGE}.sif

# 25 GiB