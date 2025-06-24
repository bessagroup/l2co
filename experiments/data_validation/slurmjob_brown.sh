#!/bin/bash
#SBATCH --job-name="DataValidation"
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --account=mbessa-condo
#SBATCH --mail-user=m.p.vanderschelling@tudelft.nl

projectfolder=$(basename $(pwd))
opt_dir="${HOME}/scratch/create_optimization_dataset_new/"
run_dir="${HOME}/scratch/${projectfolder}/${SLURM_JOB_ID}"

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
module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh 

echo "======================================="
echo " Environment Information"
echo "======================================="
echo "Loaded Modules:"
module list 2>&1  # Redirect stderr to stdout to capture module list
echo "======================================="

# Activate conda environment
unset CONDA_SHLVL
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh 
conda activate l2co

echo "Conda Environment: $(conda info --envs | grep '*' | awk '{print $1}')"

echo "======================================="
echo " Execution Information"
echo "======================================="
echo "Project Folder:   $projectfolder"
echo "Run Directory:    $run_dir"
echo "Executing: python main.py ++hydra.run.dir=$run_dir $@"
echo "======================================="

# Run the Python script
python main.py ++hydra.run.dir=$run_dir ++opt_dir=$opt_dir "$@"
