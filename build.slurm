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
DOCKER_URL="docker://thewillyp/${IMAGE}:master-1.0.6@sha256:de1f866d7715e111c3bb667b284ee41de527f38e705fe140166dbb65d0e1e7ee"
SAVE_DIR="/scratch/${USER}/"


singularity build --force /scratch/${USER}/images/${IMAGE}.sif ${DOCKER_URL} 
singularity overlay create --size 10240 /scratch/${USER}/images/${IMAGE}.sif
