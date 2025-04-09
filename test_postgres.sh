#!/bin/bash
#SBATCH --job-name=test_psql         # Job name
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1                   # Number of tasks
#SBATCH --cpus-per-task=1            # Number of CPU cores per task
#SBATCH --mem=2G                     # Memory per node
#SBATCH --time=00:05:00              # Time limit (5 minutes)
#SBATCH --output=test_psql_%j.out    # Output file (%j is job ID)
#SBATCH --error=test_psql_%j.err     # Error file (%j is job ID)

# Run psql with PGPASSWORD set as an environment variable
# singularity exec  --containall --cleanenv --env PGPASSWORD=march-underuse-buffoon-cupped-mutable docker://postgres:17 psql -h cm001 -p 5432 -U wlp9800 -d sweeps -c "SELECT * FROM test_table;" -w

singularity exec --containall --cleanenv --bind /scratch/wlp9800/postgres-test:/var \
    docker://postgres:17 pg_ctl stop
echo "Postgres shutdown complete."