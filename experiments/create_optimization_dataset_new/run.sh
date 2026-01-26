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

if [[ $1 == project_job=* ]]; then
    # Extract the ID after the "=" sign (e.g., 123456)
    project_job="${1#*=}"
    echo "Resuming existing project. Running: sbatch $JOB_CREATE project_job=$project_job"
    id_create=$(sbatch $JOB_CREATE project_job=$project_job | awk '{print $4}')
    
else
    # Case: Fresh start. Capture the ID from the new sbatch call
    echo "Started new project with Job ID: $id_create"
    id_create=$(sbatch $JOB_CREATE | awk '{print $4}')
    project_job=$id_create
fi

echo "Waiting for Job $id_create to complete..."

while true; do
    # Get the status of the job
    job_status=$(sacct -j "$id_create" --format=State --noheader | head -n 1 | awk '{print $1}')

    if [[ "$job_status" == "COMPLETED" ]]; then
        echo "Job $id_create finished successfully."
        break
    elif [[ "$job_status" == "FAILED" || "$job_status" == "CANCELLED" || "$job_status" == "NODE_FAIL" || "$job_status" == "TIMEOUT" ]]; then
        echo "Error: Job $id_create ended with status: $job_status"
        exit 1
    elif [[ -z "$job_status" ]]; then
        echo "Job $id_create not yet found in sacct... waiting."
    else
        echo "Job $id_create status: $job_status. Checking again in 5 seconds..."
    fi

    sleep 5
done

projectfolder=$(basename $(pwd))
jobs_file="${HOME}/scratch/${projectfolder}/${project_job}/experiment_data/jobs.csv"

# Double check file existence before grepping
if [ ! -f "$jobs_file" ]; then
    echo "Error: Job $id_create reported COMPLETED, but $jobs_file is missing!"
    exit 1
fi

# Count exactly how many lines have the status 'OPEN'
num_jobs=$(grep -c "OPEN" "$jobs_file")
# num_jobs=$(($(wc -l < "$jobs_file") - 2))
num_jobs=$(( num_jobs > 900 ? 900 : num_jobs ))


# echo number of jobs
echo "Number of jobs to run: $num_jobs"

id_run=$(sbatch --array=0-$(($num_jobs-1))%64 -o "slurm-${project_job}.out"  --dependency=afterok:$id_create --export=id_create=$id_create $JOB_RUN  ++project_job=$project_job | awk '{print $4}')

id_post=$(sbatch -o "slurm-${project_job}.out" --dependency=afterany:$id_run --export=id_create=$id_create $JOB_POST ++project_job=$project_job | awk '{print $4}')

# Job ID table
echo -e "\nCREATE DATASET\n------------------------------------------------------"
echo -e "Job ID\t\t\tDescription\n------------------------------------------------------"
echo -e "$id_create\tFirst Job: Creating the ExperimentData"
echo -e "$id_run\tSecond Job: Running the experiments"
echo -e "$id_post\tThird Job: Post-processing the experiments"
echo -e "------------------------------------------------------"
