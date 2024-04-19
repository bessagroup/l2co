#!/bin/bash
#SBATCH --job-name="CreateTrainingData"
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=25GB
#SBATCH --account=innovation
#SBATCH --partition=compute
#SBATCH --mail-user=m.p.vanderschelling@tudelft.nl

# 2. Execute
# ============================================

JOB_ID=$(echo "${SLURM_ARRAY_JOB_ID}" | sed 's/\[[^][]*\]//g')

module load miniconda3
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate f3dasm_env

if ! [ -n "${SLURM_ARRAY_TASK_ID+1}" ]; then
  SLURM_ARRAY_TASK_ID=-1
fi


python main.py ++hpc.jobid=${SLURM_ARRAY_TASK_ID} \
 hydra.run.dir=/scratch/mpvanderschell/create_trainingdata_two_optimizers/${now:%Y-%m-%d}/${JOB_ID}
