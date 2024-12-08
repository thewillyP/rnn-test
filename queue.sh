#!/bin/bash
#SBATCH --job-name=job_array
#SBATCH --output=job_array_%A_%a.out
#SBATCH --error=job_array_%A_%a.err
#SBATCH --time=12:00:00  # 8 hours
#SBATCH --ntasks=1       # 1 CPU core
#SBATCH --mem=1M         # 1 MB of memory

# Set the start, end, and increment values
start=1
end=6400
increment=1500

# Loop over the range with the specified increment
for (( i=$start; i<=$end; i+=$increment ))
do
    # Calculate the end of the current array range
    # Ensure the last range doesn't exceed the total
    range_end=$(($i + $increment - 1))
    if [ $range_end -gt $end ]; then
        range_end=$end
    fi

    # Submit a job array for the current range
    WANDB_API_KEY=${WANDB_API_KEY} sbatch --array=$i-$range_end --wait run.slurm
done
