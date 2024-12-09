#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=01:30:00
#SBATCH --job-name=rnn_experiments
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --array=1
#SBATCH --output=test.out
#SBATCH --error=test.err
#SBATCH --mail-type=END
#SBATCH --mail-user=wlp9800@nyu.edu


CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID))p" configs_test.txt)


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
python /rnn-test/src/main.py '"${CONFIG}"''
