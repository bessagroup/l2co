#!/bin/bash
#SBATCH --job-name="RunStage1Training"
#SBATCH --partition=batch
#SBATCH --time=00:30:00
#SBATCH --account=default
#SBATCH --mail-user=m.p.vanderschelling@tudelft.nl
#SBATCH --mem=8GB

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

# 2. Load LaTeX module and resolve paths
module load texlive

# Find the binary path and the specific distribution tree
LATEX_BIN_PATH=$(dirname $(which pdflatex))
TEXMF_PATH=$(kpsewhich -var-value=TEXMFDIST)

# The "Atomic" Bind: Find the shared root directory of the entire distribution
# This usually moves up 2-3 levels from /bin to catch /texmf-dist, /fonts, /texmf-var, etc.
TEXLIVE_BASE=$(python3 -c "import os; print(os.path.abspath('$LATEX_BIN_PATH/../../..'))")

echo "======================================="
echo " Environment Information"
echo "======================================="
echo "Loaded Modules:"
module list 2>&1
echo "LaTeX Bin Path:   $LATEX_BIN_PATH"
echo "TEXLIVE Base:     $TEXLIVE_BASE"
echo "TEXMF Dist Path:  $TEXMF_PATH"
echo "======================================="

# 3. Configure Apptainer Paths & Environment
# Bind host directories + the entire TeX distribution tree
export APPTAINER_BINDPATH="/oscar/home/$USER,/oscar/scratch/$USER,/oscar/data,$TEXLIVE_BASE"

# Inject LaTeX variables into the container so kpathsea knows where to look
export APPTAINERENV_APPEND_PATH="$LATEX_BIN_PATH"
export APPTAINERENV_TEXMFDIST="$TEXMF_PATH"

# Setup writable Matplotlib/LaTeX cache to avoid permission errors
export MPLCONFIGDIR="/oscar/scratch/$USER/.matplotlib"
mkdir -p $MPLCONFIGDIR

# 4. Define Container and Script variables
CONTAINER=/users/$USER/jax312.simg
SCRIPT=main_run.py

# Get project folders
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