#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=12G
#SBATCH --time=06:00:00
#SBATCH --job-name=run
#SBATCH --error=run.err
#SBATCH --output=run.log
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=wlp9800@nyu.edu

# Changes often
USER=wlp9800
IMAGE="rnn-test"
CONDA_ENV=master



# 1. check if key exist 
# 2. trust our own key 
# 3. then will copy this key into container through overlay, makes container trust our key. No need to rebuild docker with key injected bc inject w/ overlay
# 4. let's us jump from greene -> container w/ no password
grep -qxFf $HOME/.ssh/id_rsa.pub $HOME/.ssh/authorized_keys || cat $HOME/.ssh/id_rsa.pub >> $HOME/.ssh/authorized_keys  # 1. 2.



singularity exec --nv --containall --no-home --cleanenv \
--overlay /scratch/${USER}/${IMAGE}.ext3:rw \
--bind /home/${USER}/.ssh,/home/${USER}/dev \
/scratch/${USER}/images/${IMAGE}.sif bash -c \
'micromamba shell init --shell bash --root-prefix=/opt/conda && \
grep -q "source /.singularity.d/env/10-docker2singularity.sh" /home/'"${USER}"'/.bashrc || echo "source /.singularity.d/env/10-docker2singularity.sh" >> /home/'"${USER}"'/.bashrc && \
grep -q "export LD_LIBRARY_PATH=\"/.singularity.d/libs\"" /home/'"${USER}"'/.bashrc || echo "export LD_LIBRARY_PATH=\"/.singularity.d/libs\"" >> /home/'"${USER}"'/.bashrc && \
source /home/'"${USER}"'/.bashrc && \
micromamba activate '"${CONDA_ENV}"' && \
mkdir -p /home/'"${USER}"'/notebooks && \
jupyter lab --notebook-dir=/home/'"${USER}"'/notebooks --ip="0.0.0.0" --port=8888 --no-browser & \
mkdir -p /home/'"${USER}"'/hostkeys && \
ssh-keygen -q -N "" -t rsa -b 4096 -f /home/'"${USER}"'/hostkeys/ssh_host_rsa_key <<< y && \
/usr/sbin/sshd -D -p 2222 -o UsePAM=no -h /home/'"${USER}"'/hostkeys/ssh_host_rsa_key'

# sshd must be the last cmd to run to be a blocking process
