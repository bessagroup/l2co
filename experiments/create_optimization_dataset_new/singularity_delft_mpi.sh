#!/bin/bash
#SBATCH --job-name="RunTaskOnSchedule"
#SBATCH --partition=compute
#SBATCH --time=24:00:00
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2GB
#SBATCH --account=research-ME-mse

echo $LD_LIBRARY_PATH
unset LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH

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

module load 2025
module load openmpi

echo "======================================="
echo " Environment Information"
echo "======================================="
echo "Loaded Modules:"
module list 2>&1  # Redirect stderr to stdout to capture module list
echo "======================================="

export APPTAINER_BINDPATH="/scratch/$USER,/home/$USER"
CONTAINER=/home/$USER/jax312.simg
SCRIPT=main.py

model_dir="/scratch/$USER/train_model/"
project_dir="/scatch/$USER/create_optimization_dataset_new/"


# Get project folder
projectfolder=$(basename $(pwd))
run_dir="/scratch/$USER/${projectfolder}/${SLURM_JOB_ID}"

echo "======================================="
echo " Execution Information"
echo "======================================="
echo "Project Folder:   $projectfolder"
echo "Run Directory:    $run_dir"
echo "Executing: srun --mpi=pmix apptainer exec $CONTAINER python3 -u $SCRIPT ++hydra.run.dir=$run_dir ++model_dir=$model_dir ++project_dir=$project_dir $@" 
echo "======================================="

export JAX_PLATFORM_NAME=cpu

# Run the Python script with MPI
# singularity exec $CONTAINER python3 -u $SCRIPT ++hydra.run.dir=$run_dir $@
srun --mpi=pmix apptainer exec $CONTAINER python3 -u $SCRIPT ++hydra.run.dir=$run_dir ++model_dir=$model_dir ++project_dir=$project_dir $@