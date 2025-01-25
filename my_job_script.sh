#!/bin/bash
#SBATCH --ntasks=1       # 1 CPU core
#SBATCH --mem=1M         # 1 MB of memory
#SBATCH --time=00:05:00  # 5 minutes, just for testing

# Output the job ID and task ID for verification
echo "Job ID: $SLURM_JOB_ID"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"

# Wait for 2 seconds
sleep 2
