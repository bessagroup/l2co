#!/bin/bash
#SBATCH --job-name="RunTaskOnSchedule"
#SBATCH --time=12:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4GB
#SBATCH --account=research-ME-mse
#SBATCH --partition=compute
#SBATCH --mail-user=m.p.vanderschelling@tudelft.nl

# 2. Execute
# ============================================

# Load necessary modules
module load 2023r1
module load openmpi
module load miniconda3

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Activate conda environment
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate f3dasm_env

# Handle Slurm Array Job ID
if ! [ -n "${SLURM_ARRAY_TASK_ID+1}" ]; then
  SLURM_ARRAY_TASK_ID=-1
  SLURM_ARRAY_JOB_ID="${SLURM_JOB_ID}"
fi

projectfolder=$(basename $(pwd))

srun python main.py hydra.run.dir=/scratch/${USER}/${projectfolder}/${SLURM_ARRAY_JOB_ID}
