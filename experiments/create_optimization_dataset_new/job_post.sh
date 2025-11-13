#!/bin/bash
#SBATCH --job-name="PostTaskOnSchedule"
#SBATCH --partition=batch
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2GB
#SBATCH --account=mbessa-condo
#SBATCH --mail-user=m.p.vanderschelling@tudelft.nl

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
SCRIPT=main_post.py

model_dir="${HOME}/scratch/train_model/"
project_dir="${HOME}/scratch/create_optimization_dataset_new/"

# Get project folder
projectfolder=$(basename $(pwd))
run_dir="${HOME}/scratch/${projectfolder}/${SLURM_JOB_ID}"

echo "======================================="
echo " Execution Information"
echo "======================================="
echo "Project Folder:   $projectfolder"
echo "Run Directory:    $run_dir"
echo "Executing: srun apptainer exec $CONTAINER python3 -u $SCRIPT ++hydra.run.dir=$run_dir ++model_dir=$model_dir ++project_dir=$project_dir $@" 
echo "======================================="

export JAX_PLATFORM_NAME=cpu

# Run the Python script with MPI
srun apptainer exec $CONTAINER python3 -u $SCRIPT ++hydra.run.dir=$run_dir ++model_dir=$model_dir ++project_dir=$project_dir $@