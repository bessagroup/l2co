#!/bin/bash
#SBATCH --job-name="l2co_main"
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=batch
#SBATCH --account=default
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Usage: sbatch submit_job.sh experiment=<name> [hydra_overrides...]
#
# Submits a single low-resource job that cd's into the experiment directory
# and runs: uv run main.py cluster=brown ++rootdir=/oscar/scratch/$USER/<experiment>
#
# Additional Hydra overrides are forwarded to main.py, e.g.:
#   sbatch submit_job.sh experiment=online_rl "++rootdir=/oscar/scratch/$USER/evaluate_rl"

experiment=""
extra_args=()

for arg in "$@"; do
    case $arg in
        experiment=*) experiment="${arg#*=}" ;;
        *) extra_args+=("$arg") ;;
    esac
done

if [[ -z "$experiment" ]]; then
    echo "ERROR: 'experiment' argument is required."
    echo "Usage: sbatch $0 experiment=<name> [hydra_overrides...]"
    exit 1
fi

# REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
ROOTDIR="/oscar/scratch/$USER/$experiment"

export HYDRA_FULL_ERROR=1

echo "======================================="
echo " Job: $SLURM_JOB_ID"
echo " Experiment: $experiment"
echo " Root dir: $ROOTDIR"
echo " Extra Hydra overrides: ${extra_args[*]}"
echo "======================================="

cd "$experiment"
uv run main.py cluster=brown ++mode=slurm ++rootdir=$ROOTDIR "${extra_args[@]}"
