#!/bin/bash

# Define variables
tasks="default_task"          # Assign appropriate value
JOB_SCRIPT="slurmjob_brown.sh"
DIR_CREATE_OPT_DATA="/users/mvander7/L2CO/experiments/create_optimization_dataset_new"
DIR_VALIDATION="/users/mvander7/L2CO/experiments/data_validation"
DIR_TRAIN_MODEL="/users/mvander7/L2CO/experiments/train_model"

# First job - Run all tasks with combination schedules
jobID_1_combination=$(cd $DIR_CREATE_OPT_DATA && sbatch $JOB_SCRIPT schedules=combination tasks=$tasks | cut -f 4 -d' ')

# Second job - Run all tasks with single schedules
jobID_2_single=$(cd $DIR_CREATE_OPT_DATA && sbatch $JOB_SCRIPT schedules=single tasks=$tasks | cut -f 4 -d' ')

# Third job - Data validation on single schedules
jobID_3=$(cd $DIR_VALIDATION && sbatch --dependency=afterok:$jobID_2_single $JOB_SCRIPT +path=$jobID_2_single | cut -f 4 -d' ')

# Fourth job - Data validation on combination schedules
jobID_4=$(cd $DIR_VALIDATION && sbatch --dependency=afterok:$jobID_1_combination $JOB_SCRIPT +path=$jobID_1_combination | cut -f 4 -d' ')

# Fifth job - Train L2CO model
jobID_5=$(cd $DIR_TRAIN_MODEL && sbatch --dependency=afterany:$jobID_4 $JOB_SCRIPT +path=$jobID_1_combination | cut -f 4 -d' ')

# Sixth job - Run L2CO on the tasks
jobID_6=$(cd $DIR_CREATE_OPT_DATA && sbatch --dependency=afterany:$jobID_5 $JOB_SCRIPT schedules=l2co tasks=$tasks +path=$jobID_5 | cut -f 4 -d' ')

# Seventh job - Data validation on L2CO results
cd $DIR_VALIDATION && sbatch --dependency=afterok:$jobID_6 $JOB_SCRIPT +path=$jobID_6
