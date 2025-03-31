#!/bin/bash
#SBATCH --job-name="RunTaskOnSchedule"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --nodes=1-5
#SBATCH --cpus-per-task=4
#SBATCH --account=mbessa-condo
#SBATCH --mail-user=m.p.vanderschelling@tudelft.nl

echo $LD_LIBRARY_PATH
unset LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH

echo "======================================="
echo " Slurm GPU Job Information"
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
module load cuda cudnn
module load hpcx-mpi
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

# Handle Slurm Array Job ID
if ! [ -n "${SLURM_ARRAY_TASK_ID+1}" ]; then
  SLURM_ARRAY_TASK_ID=-1
  SLURM_ARRAY_JOB_ID="${SLURM_JOB_ID}"
fi

model_dir="${HOME}/scratch/train_model/"


# Get project folder
projectfolder=$(basename $(pwd))
run_dir="${HOME}/scratch/${projectfolder}/${SLURM_ARRAY_JOB_ID}"

echo "======================================="
echo " Execution Information"
echo "======================================="
echo "Project Folder:   $projectfolder"
echo "Run Directory:    $run_dir"
echo "Executing: mpirun python main.py ++hydra.run.dir=$run_dir ++model_dir=$model_dir $@" 
echo "======================================="

# Run the Python script with MPI
python main.py ++hydra.run.dir=$run_dir ++model_dir=$model_dir +mode=cluster "$@"