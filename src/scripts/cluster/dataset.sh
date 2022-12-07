#!/usr/bin/env bash
#SBATCH -A project01854
#SBATCH -J hex_pg
#SBATCH --mail-type=NONE
#SBATCH -a 1-6%6
#SBATCH -n 1
#SBATCH -c 3
#SBATCH --mem-per-cpu=3800
#SBATCH -t 23:58:00
#SBATCH -o /home/jb66zuhe/MasterThesis/src/scripts/log/HEX_DATA_%A_%a-out.txt
#SBATCH -e /home/jb66zuhe/MasterThesis/src/scripts/log/error/%A_%a-err.txt
###############################################################################

# Setup modules
module purge

# Set ulimit
ulimit -n 16384

#module load gcc cuda
#nvidia-smi

# Your PROGRAM call starts here
echo "Starting Job $SLURM_JOB_ID, Index $SLURM_ARRAY_TASK_ID"

# Activate the anaconda environment
eval "$($HOME/anaconda3/bin/conda shell.bash hook)"
conda activate rl

# Move to directory
THESIS_DIR="$HOME/MasterThesis"
cd "$THESIS_DIR"

# Start script
python -m src.scripts.pg --log_to_writer True --log_to_file True --log_name ppo_train.txt
