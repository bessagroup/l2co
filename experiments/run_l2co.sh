#!/bin/sh

#SBATCH --job-name="C_RunL2CO"
#SBATCH -N 1
#SBATCH --time=00:30:00
#SBATCH --partition=batch
#SBATCH -n 1
#SBATCH --account=mbessa-condo

# Parse command-line arguments with defaults
tasks_path="${tasks_path:?Error: tasks_path not set. Exiting...}"
tasks="continue"
schedules="l2co"
budgets="${budgets:?Error: budgets not set. Exiting...}"
optimizers="l2co"
samplers="${samplers:-random}"
seed="${seed:-42}"
path="${path:-.}"
mode="${mode:-cpu}"

# Job name variables
JOB_NAME_1="R_l2co_$tasks_path"
JOB_NAME_2="V_l2co_$tasks_path"

# Set job script based on mode
JOB_SCRIPT="slurmjob_brown.sh"
[ "$mode" = "gpu" ] && JOB_SCRIPT="slurmjob_brown_gpu.sh"

JOB_SCRIPT_VALIDATION="slurmjob_brown.sh"

# Debugging output
echo -e "TASKS: $tasks\nOPTIMIZERS: $optimizers\nSCHEDULES: $schedules\nSEED: $seed\nBUDGETS: $budgets\nSAMPLERS: $samplers\nPATH: $path\nMODE: $mode"

# Directories and job script
BASE_DIR="/users/mvander7/L2CO/experiments"
DIR_CREATE_OPT_DATA="$BASE_DIR/create_optimization_dataset_new"
DIR_VALIDATION="$BASE_DIR/data_validation"

# Submit first job
jobID_1=$(cd $DIR_CREATE_OPT_DATA && sbatch --job-name=$JOB_NAME_1 $JOB_SCRIPT schedules=$schedules tasks=$tasks optimizers=$optimizers budgets=$budgets samplers=$samplers +path=$path +tasks.path=$tasks_path ++seed=$seed | awk '{print $4}')
echo "First Job submitted: $jobID_1"

# Submit second job (dependent on first job)
jobID_2=$(cd $DIR_VALIDATION && sbatch --job-name=$JOB_NAME_2 --dependency=afterany:$jobID_1 $JOB_SCRIPT_VALIDATION +path=$jobID_1 | awk '{print $4}')
echo "Second Job submitted: $jobID_2"

# Job ID table
echo -e "\nRUN L2CO\n------------------------------------------------------"
echo -e "Job ID\t\t\tDescription\n------------------------------------------------------"
echo -e "$jobID_1\tFirst Job: Running all tasks with schedules"
echo -e "$jobID_2\tSecond Job: Data validation"
echo -e "------------------------------------------------------"
