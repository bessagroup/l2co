#!/bin/bash
#SBATCH --job-name="CreateTrainingData"
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --account=mbessa-condo
#SBATCH --partition=batch
#SBATCH --mail-user=m.p.vanderschelling@tudelft.nl

# 2. Execute
# ============================================

module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh 

conda activate f3dasm_env

if ! [ -n "${SLURM_ARRAY_TASK_ID+1}" ]; then
  SLURM_ARRAY_TASK_ID=-1
  SLURM_ARRAY_JOB_ID="${SLURM_JOB_ID}"
fi

projectfolder=$(basename $(pwd))

python main.py ++hpc.jobid=${SLURM_ARRAY_TASK_ID} \
 hydra.run.dir=/scratch/${USER}/${projectfolder}/${SLURM_ARRAY_JOB_ID}
