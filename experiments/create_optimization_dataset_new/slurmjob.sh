#!/bin/bash
#SBATCH --job-name="RunTaskOnSchedule"
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=64
#SBATCH --nodes=1-3
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --account=research-ME-mse
#SBATCH --mail-user=m.p.vanderschelling@tudelft.nl

echo "======================================="
echo " Slurm Job Information"
echo "======================================="
echo "Job ID:          $SLURM_JOB_ID"
echo "Job Name:        $SLURM_JOB_NAME"
echo "Partition:       $SLURM_JOB_PARTITION"
echo "Nodes Allocated: $SLURM_JOB_NODELIST"
echo "Total Tasks:     $SLURM_NTASKS"
echo "Tasks per Node:  $SLURM_NTASKS_PER_NODE"
echo "CPUs per Task:   $SLURM_CPUS_PER_TASK"
echo "Memory per CPU:  $SLURM_MEM_PER_CPU"
echo "======================================="

# Load necessary modules
module load 2023r1
module load openmpi
module load miniconda3

echo "======================================="
echo " Environment Information"
echo "======================================="
echo "Loaded Modules:"
module list 2>&1  # Redirect stderr to stdout to capture module list
echo "OpenMP Threads:   $OMP_NUM_THREADS"
echo "======================================="

# Activate conda environment
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate l2co
echo "Conda Environment: $(conda info --envs | grep '*' | awk '{print $1}')"

# Handle Slurm Array Job ID
if ! [ -n "${SLURM_ARRAY_TASK_ID+1}" ]; then
  SLURM_ARRAY_TASK_ID=-1
  SLURM_ARRAY_JOB_ID="${SLURM_JOB_ID}"
fi

model_dir="/scratch/${USER}/train_model/"
project_dir="/scratch/${USER}/create_optimization_dataset_new/"

# Get project folder
projectfolder=$(basename $(pwd))
run_dir="/scratch/${USER}/${projectfolder}/${SLURM_ARRAY_JOB_ID}"

echo "======================================="
echo " Execution Information"
echo "======================================="
echo "Project Folder:   $projectfolder"
echo "Run Directory:    $run_dir"
echo "Executing: srun python main.py ++hydra.run.dir=$run_dir  ++model_dir=$model_dir ++project_dir=$project_dir $@"
echo "======================================="

# Run the Python script with MPI
srun python main.py ++hydra.run.dir=$run_dir ++model_dir=$model_dir ++project_dir=$project_dir "$@"
