#!/bin/bash
#SBATCH --job-name="Run"
#SBATCH -N 1
#SBATCH --time=00:30:00
#SBATCH --partition=batch
#SBATCH -n 1
#SBATCH --account=default

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

JOB_CREATE="job_create.sh"
JOB_RUN="job_run.sh"
JOB_POST="job_post.sh"


id_create=$(sbatch $JOB_CREATE | awk '{print $4}')

projectfolder=$(basename $(pwd))
jobs_file="${HOME}/scratch/${projectfolder}/${id_create}/experiment_data/jobs.csv"

# Count number of lines in jobs_file and subtract 2 (header + 0-indexing)
num_jobs=$(($(wc -l < "$jobs_file") - 2))

# echo number of jobs
echo "Number of jobs to run: $num_jobs"

id_run=$(sbatch --array=0-${num_jobs}%16 -o "slurm-${id_create}.out"  --dependency=afterok:$id_create --export=id_create=$id_create $JOB_RUN  ++project_job=$id_create | awk '{print $4}')

id_post=$(sbatch -o "slurm-${id_create}.out" --dependency=afterany:$id_run --export=id_create=$id_create $JOB_POST ++project_job=$id_create | awk '{print $4}')

# Job ID table
echo -e "\nCREATE DATASET\n------------------------------------------------------"
echo -e "Job ID\t\t\tDescription\n------------------------------------------------------"
echo -e "$id_create\tFirst Job: Creating the ExperimentData"
echo -e "$id_run\tSecond Job: Running the experiments"
echo -e "$id_post\tThird Job: Post-processing the experiments"
echo -e "------------------------------------------------------"
