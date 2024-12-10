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
DOCKER_URL="docker://thewillyp/${IMAGE}:master-1.0.22@sha256:4b89f533f347a7f295c219e786a7d48dde5fefaf165d814abf71ffbbf9514b31"
SAVE_DIR="/scratch/${USER}/"


singularity build --force /scratch/${USER}/images/${IMAGE}.sif ${DOCKER_URL} 
singularity overlay create --size 25600 /scratch/${USER}/images/${IMAGE}.sif

# 25 GiB