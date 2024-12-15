#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --job-name=rnn_experiments
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --array=1-4
#SBATCH --output="/scratch/wlp9800/logs/%x-%A-%a.out"
#SBATCH --error="/scratch/wlp9800/logs/%x-%A-%a.err"

CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID))p" configs.txt)


# Changes often
USER=wlp9800
IMAGE="rnn-test"
CONDA_ENV=master


singularity exec --nv --containall --no-home --cleanenv --writable-tmpfs \
--env WANDB_API_KEY=${WANDB_API_KEY} \
/scratch/${USER}/images/${IMAGE}.sif bash -c \
'micromamba shell init --shell bash --root-prefix=/opt/conda && \
grep -q "source /.singularity.d/env/10-docker2singularity.sh" /home/'"${USER}"'/.bashrc || echo "source /.singularity.d/env/10-docker2singularity.sh" >> /home/'"${USER}"'/.bashrc && \
grep -q "export LD_LIBRARY_PATH=\"/.singularity.d/libs\"" /home/'"${USER}"'/.bashrc || echo "export LD_LIBRARY_PATH=\"/.singularity.d/libs\"" >> /home/'"${USER}"'/.bashrc && \
source /home/'"${USER}"'/.bashrc && \
micromamba activate '"${CONDA_ENV}"' && \
cd /rnn-test && \
git config --global --add safe.directory /rnn-test && \
python /rnn-test/src/main.py '"${CONFIG}"''
