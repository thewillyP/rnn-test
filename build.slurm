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
DOCKER_URL="docker://thewillyp/${IMAGE}:master-1.0.9@sha256:e42f481210ed756fd2931c7c398fde6c5e406d2f042b878618776dc750bd65bc"
SAVE_DIR="/scratch/${USER}/"


singularity build --force /scratch/${USER}/images/${IMAGE}.sif ${DOCKER_URL} 
singularity overlay create --size 10240 /scratch/${USER}/images/${IMAGE}.sif
