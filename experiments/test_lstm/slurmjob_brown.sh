#!/bin/bash
#SBATCH --job-name="test-lstm"
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-user=m.p.vanderschelling@tudelft.nl

# 2. Execute
# ============================================


echo $LD_LIBRARY_PATH
unset LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH

source /users/$USER/jax.venv/bin/activate
python3 -u main.py