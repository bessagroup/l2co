#!/bin/sh

#SBATCH --job-name="CombinedScript"
#SBATCH -N 1
#SBATCH --time=00:30:00
#SBATCH --partition=batch
#SBATCH -n 1
#SBATCH --account=mbessa-condo

# Job name variables
JOB_NAME_1="R_comb_1"
JOB_NAME_2="R_sing_2"
JOB_NAME_3="V_single_3"
JOB_NAME_4="V_comb_4"
JOB_NAME_5="T_5"
JOB_NAME_6="R_l2co_6"
JOB_NAME_7="V_l2co_7"

# Parse command line arguments passed to sbatch via environment variables
if [ ! -z "$SLURM_TASKS" ]; then
    tasks="$SLURM_TASKS"  # Assign the value passed with --export to the 'tasks' variable
fi

# Define other variables
JOB_SCRIPT="slurmjob_brown.sh"
DIR_CREATE_OPT_DATA="/users/mvander7/L2CO/experiments/create_optimization_dataset_new"
DIR_VALIDATION="/users/mvander7/L2CO/experiments/data_validation"
DIR_TRAIN_MODEL="/users/mvander7/L2CO/experiments/train_model"

# First job - Run all tasks with combination schedules
echo "Starting First Job: Running all tasks with combination schedules..."
jobID_1_combination=$(cd $DIR_CREATE_OPT_DATA && sbatch --job-name=$JOB_NAME_1 $JOB_SCRIPT schedules=combination tasks=$tasks | cut -f 4 -d' ')
echo "First Job (jobID_1_combination) submitted with job ID: $jobID_1_combination"

# Second job - Run all tasks with single schedules
echo "Starting Second Job: Running all tasks with single schedules..."
jobID_2_single=$(cd $DIR_CREATE_OPT_DATA && sbatch --job-name=$JOB_NAME_2 $JOB_SCRIPT schedules=single tasks=$tasks | cut -f 4 -d' ')
echo "Second Job (jobID_2_single) submitted with job ID: $jobID_2_single"

# Third job - Data validation on single schedules
echo "Starting Third Job: Data validation on single schedules, depends on jobID_2_single..."
jobID_3=$(cd $DIR_VALIDATION && sbatch --job-name=$JOB_NAME_3 --dependency=afterany:$jobID_2_single $JOB_SCRIPT +path=$jobID_2_single | cut -f 4 -d' ')
echo "Third Job (jobID_3) submitted with job ID: $jobID_3"

# Fourth job - Data validation on combination schedules
echo "Starting Fourth Job: Data validation on combination schedules, depends on jobID_1_combination..."
jobID_4=$(cd $DIR_VALIDATION && sbatch --job-name=$JOB_NAME_4 --dependency=afterany:$jobID_1_combination $JOB_SCRIPT +path=$jobID_1_combination | cut -f 4 -d' ')
echo "Fourth Job (jobID_4) submitted with job ID: $jobID_4"

# Fifth job - Train L2CO model
echo "Starting Fifth Job: Train L2CO model, depends on jobID_4..."
jobID_5=$(cd $DIR_TRAIN_MODEL && sbatch --job-name=$JOB_NAME_5 --dependency=afterok:$jobID_4 $JOB_SCRIPT +path=$jobID_1_combination | cut -f 4 -d' ')
echo "Fifth Job (jobID_5) submitted with job ID: $jobID_5"

# Sixth job - Run L2CO on the tasks
echo "Starting Sixth Job: Run L2CO on the tasks, depends on jobID_5..."
jobID_6=$(cd $DIR_CREATE_OPT_DATA && sbatch --job-name=$JOB_NAME_6 --dependency=afterok:$jobID_5 $JOB_SCRIPT schedules=l2co tasks=$tasks +path=$jobID_5 | cut -f 4 -d' ')
echo "Sixth Job (jobID_6) submitted with job ID: $jobID_6"

# Seventh job - Data validation on L2CO results
echo "Starting Seventh Job: Data validation on L2CO results, depends on jobID_6..."
cd $DIR_VALIDATION && sbatch --job-name=$JOB_NAME_7 --dependency=afterany:$jobID_6 $JOB_SCRIPT +path=$jobID_6
echo "Seventh Job (Data validation on L2CO results) submitted."

# Echo job ID table at the bottom
echo -e "\nJob IDs and Descriptions:"
echo -e "------------------------------------------------------"
echo -e "Job ID\t\t\tDescription"
echo -e "------------------------------------------------------"
echo -e "$jobID_1_combination\tFirst Job: Running all tasks with combination schedules"
echo -e "$jobID_2_single\tSecond Job: Running all tasks with single schedules"
echo -e "$jobID_3\t\tThird Job: Data validation on single schedules"
echo -e "$jobID_4\t\tFourth Job: Data validation on combination schedules"
echo -e "$jobID_5\t\tFifth Job: Train L2CO model"
echo -e "$jobID_6\t\tSixth Job: Run L2CO on the tasks"
echo -e "------------------------------------------------------"
