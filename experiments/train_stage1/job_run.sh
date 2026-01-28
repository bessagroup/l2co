#!/bin/bash
#SBATCH --job-name="RunStage1Training"
#SBATCH --partition=batch
#SBATCH --time=00:30:00
#SBATCH --account=default
#SBATCH --mail-user=m.p.vanderschelling@tudelft.nl
#SBATCH --mem=8GB

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

echo "======================================="
echo " Environment Information"
echo "======================================="
echo "Loaded Modules:"
module list 2>&1  # Redirect stderr to stdout to capture module list
echo "======================================="

export APPTAINER_BINDPATH="/oscar/home/$USER,/oscar/scratch/$USER,/oscar/data"
CONTAINER=/users/$USER/jax312.simg
SCRIPT=main_run.py

# Get project folder
projectfolder=$(basename $(pwd))

project_dir="${HOME}/scratch/${projectfolder}/"
dataset_dir="${HOME}/scratch/create_optimization_dataset_new/"
run_dir="${HOME}/scratch/${projectfolder}/${id_create}"

echo "======================================="
echo " Execution Information"
echo "======================================="
echo "Project Folder:   $projectfolder"
echo "Run Directory:    $run_dir"
echo "Executing: apptainer exec $CONTAINER python3 -u $SCRIPT ++hydra.run.dir=$run_dir ++project_dir=$project_dir ++dataset_dir=$dataset_dir ++job_number=$SLURM_ARRAY_TASK_ID $@" 
echo "======================================="

export JAX_PLATFORM_NAME=cpu
export JAX_LOG_LEVEL=error

# Run the Python script with MPI
apptainer exec $CONTAINER python3 -u $SCRIPT ++hydra.run.dir=$run_dir ++project_dir=$project_dir ++dataset_dir=$dataset_dir ++job_number=$SLURM_ARRAY_TASK_ID $@