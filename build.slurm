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
DOCKER_URL="docker://thewillyp/${IMAGE}:master-1.0.12@sha256:fb45b7428d4a0c5c8d32db0567da0cea6e04b4ebcde82722620bd0a6b3a8df70"
SAVE_DIR="/scratch/${USER}/"


singularity build --force /scratch/${USER}/images/${IMAGE}.sif ${DOCKER_URL} 
singularity overlay create --size 10240 /scratch/${USER}/images/${IMAGE}.sif
