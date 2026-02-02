#!/bin/bash
#SBATCH --job-name="RunStage0Training"
#SBATCH --partition=batch
#SBATCH --time=00:30:00
#SBATCH --account=default
#SBATCH --mail-user=m.p.vanderschelling@tudelft.nl
#SBATCH --mem=4GB

# 1. Clean up library paths
echo "Original LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
unset LD_LIBRARY_PATH

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


echo "======================================="
echo " Environment Information"
echo "======================================="
echo "Using Manual LaTeX Root: $TEXLIVE_ROOT"
echo "======================================="

# 3. Configure Apptainer Paths & Environment
# We bind the root directory where the fonts/binaries live
export APPTAINER_BINDPATH="/oscar/home/$USER,/oscar/scratch/$USER,/oscar/data,$TEXLIVE_ROOT"

# 4. Define Container and Script variables
CONTAINER=/users/$USER/jax312.simg
SCRIPT=main_run.py

# Resolve project folders
projectfolder=$(basename $(pwd))
project_dir="${HOME}/scratch/${projectfolder}/"
dataset_dir="${HOME}/scratch/create_optimization_dataset_new/"
run_dir="${HOME}/scratch/${projectfolder}/${id_create}"

# 5. JAX Configuration
export JAX_PLATFORM_NAME=cpu
export JAX_LOG_LEVEL=error

echo "======================================="
echo " Execution Information"
echo "======================================="
echo "Project Folder:   $projectfolder"
echo "Run Directory:    $run_dir"
echo "Executing: apptainer exec $CONTAINER python3 -u $SCRIPT ..."
echo "======================================="

# 6. Execute
apptainer exec $CONTAINER python3 -u $SCRIPT \
    ++hydra.run.dir=$run_dir \
    ++project_dir=$project_dir \
    ++dataset_dir=$dataset_dir \
    ++job_number=$SLURM_ARRAY_TASK_ID \
    $@