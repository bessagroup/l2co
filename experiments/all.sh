#!/bin/bash

# Default task value
tasks="default_task"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --tasks)
      tasks="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Define other variables
JOB_SCRIPT="slurmjob_brown.sh"
DIR_CREATE_OPT_DATA="/users/mvander7/L2CO/experiments/create_optimization_dataset_new"
DIR_VALIDATION="/users/mvander7/L2CO/experiments/data_validation"
DIR_TRAIN_MODEL="/users/mvander7/L2CO/experiments/train_model"

# First job - Run all tasks with combination schedules
echo "Starting First Job: Running all tasks with combination schedules..."
jobID_1_combination=$(cd $DIR_CREATE_OPT_DATA && sbatch $JOB_SCRIPT schedules=combination tasks=$tasks | cut -f 4 -d' ')
echo "First Job (jobID_1_combination) submitted with job ID: $jobID_1_combination"

# Second job - Run all tasks with single schedules
echo "Starting Second Job: Running all tasks with single schedules..."
jobID_2_single=$(cd $DIR_CREATE_OPT_DATA && sbatch $JOB_SCRIPT schedules=single tasks=$tasks | cut -f 4 -d' ')
echo "Second Job (jobID_2_single) submitted with job ID: $jobID_2_single"

# Third job - Data validation on single schedules
echo "Starting Third Job: Data validation on single schedules, depends on jobID_2_single..."
jobID_3=$(cd $DIR_VALIDATION && sbatch --dependency=afterok:$jobID_2_single $JOB_SCRIPT +path=$jobID_2_single | cut -f 4 -d' ')
echo "Third Job (jobID_3) submitted with job ID: $jobID_3"

# Fourth job - Data validation on combination schedules
echo "Starting Fourth Job: Data validation on combination schedules, depends on jobID_1_combination..."
jobID_4=$(cd $DIR_VALIDATION && sbatch --dependency=afterok:$jobID_1_combination $JOB_SCRIPT +path=$jobID_1_combination | cut -f 4 -d' ')
echo "Fourth Job (jobID_4) submitted with job ID: $jobID_4"

# Fifth job - Train L2CO model
echo "Starting Fifth Job: Train L2CO model, depends on jobID_4..."
jobID_5=$(cd $DIR_TRAIN_MODEL && sbatch --dependency=afterany:$jobID_4 $JOB_SCRIPT +path=$jobID_1_combination | cut -f 4 -d' ')
echo "Fifth Job (jobID_5) submitted with job ID: $jobID_5"

# Sixth job - Run L2CO on the tasks
echo "Starting Sixth Job: Run L2CO on the tasks, depends on jobID_5..."
jobID_6=$(cd $DIR_CREATE_OPT_DATA && sbatch --dependency=afterany:$jobID_5 $JOB_SCRIPT schedules=l2co tasks=$tasks +path=$jobID_5 | cut -f 4 -d' ')
echo "Sixth Job (jobID_6) submitted with job ID: $jobID_6"

# Seventh job - Data validation on L2CO results
echo "Starting Seventh Job: Data validation on L2CO results, depends on jobID_6..."
cd $DIR_VALIDATION && sbatch --dependency=afterok:$jobID_6 $JOB_SCRIPT +path=$jobID_6
echo "Seventh Job (Data validation on L2CO results) submitted."
