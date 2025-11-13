#!/bin/bash
#SBATCH --job-name="Run"
#SBATCH -N 1
#SBATCH --time=00:30:00
#SBATCH --partition=batch
#SBATCH -n 1
#SBATCH --account=mbessa-condo

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

echo "======================================="
echo " Environment Information"
echo "======================================="
echo "Loaded Modules:"
module list 2>&1  # Redirect stderr to stdout to capture module list
echo "======================================="

JOB_CREATE="job_create"
JOB_RUN="job_run"
JOB_POST="job_post"

model_dir="${HOME}/scratch/train_model/"
project_dir="${HOME}/scratch/create_optimization_dataset_new/"

job_create=$(sbatch $JOB_CREATE | awk '{print $4}')

job_run=$(sbatch $JOB_CREATE --array=0-16 --afterok:$job_create ++project_job=$job_create | awk '{print $4}')

job_post=$(sbatch $JOB_POST --afterany:$job_run ++project_job=$job_create | awk '{print $4}')

# Job ID table
echo -e "\nCREATE DATASET\n------------------------------------------------------"
echo -e "Job ID\t\t\tDescription\n------------------------------------------------------"
echo -e "$job_run\tFirst Job: Creating the ExperimentData"
echo -e "$job_post\tSecond Job: Running the experiments"
echo -e "$job_post\tThird Job: Post-processing the experiments"
echo -e "------------------------------------------------------"
