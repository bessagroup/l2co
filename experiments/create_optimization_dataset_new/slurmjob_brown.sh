#!/bin/bash
#SBATCH --job-name="RunTaskOnSchedule"
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --account=mbessa-condo
#SBATCH --partition=batch
#SBATCH --mail-user=m.p.vanderschelling@tudelft.nl

# 2. Execute
# ============================================

module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh 

conda activate l2co

if ! [ -n "${SLURM_ARRAY_TASK_ID+1}" ]; then
  SLURM_ARRAY_TASK_ID=-1
  SLURM_ARRAY_JOB_ID="${SLURM_JOB_ID}"
fi

projectfolder=$(basename $(pwd))

python main.py ++hpc.jobid=${SLURM_ARRAY_TASK_ID} \
 hydra.run.dir=${HOME}/scratch/${projectfolder}/${SLURM_ARRAY_JOB_ID}
