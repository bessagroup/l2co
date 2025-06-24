#!/bin/sh

#SBATCH --job-name="C_Train"
#SBATCH -N 1
#SBATCH --time=00:30:00
#SBATCH --partition=batch
#SBATCH -n 1
#SBATCH --account=mbessa-condo

# Job name variables
JOB_NAME_1="T_l2co"

# Parse command line arguments passed to sbatch via --export
if [ ! -z "$PATH" ]; then
    path="$PATH"  # Assign the value passed with --export to the 'tasks' variable
else
    echo "Error: PATH not set. Exiting..."
    exit 1
fi

if [ ! -z "$SEED" ]; then
    seed="$SEED"  # Assign the value passed with --export to the 'seed' variable
else
    seed="42"  # Fallback default seed
fi

# Echo values for debugging
echo "PATH: $path"
echo "SEED: $seed"


# Define other variables
JOB_SCRIPT="slurmjob_brown.sh"
DIR_TRAIN_MODEL="/users/mvander7/L2CO/experiments/train_model"

# Fifth job - Train L2CO model
echo "Starting Fifth Job: Train L2CO model, depends on jobID_4..."
jobID_5=$(cd $DIR_TRAIN_MODEL && sbatch  --job-name=$JOB_NAME_1 +path=$path ++seed=$seed | cut -f 4 -d' ')
echo "Fifth Job (jobID_5) submitted with job ID: $jobID_5"

# Echo job ID table at the bottom
echo -e "TRAIN L2CO MODEL"
echo -e "\nJob IDs and Descriptions:"
echo -e "------------------------------------------------------"
echo -e "Job ID\t\t\tDescription"
echo -e "------------------------------------------------------"
echo -e "$jobID_5\t\First job: Train L2CO model"
echo -e "------------------------------------------------------"
