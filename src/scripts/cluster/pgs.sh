#!/usr/bin/env bash
#SBATCH -A project01854
#SBATCH -J pgs_pdlm
#SBATCH --mail-type=NONE
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem-per-cpu=3800
#SBATCH -t 03:00:00
#SBATCH -o /home/jb66zuhe/MasterThesis/src/scripts/log/PGS_TIME_%A_%a-out.txt
#SBATCH -e /home/jb66zuhe/MasterThesis/src/scripts/log/error/%A_%a-err.txt
###############################################################################
##SBATCH--gres=gpu:v100

# Setup modules
module purge

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
python -m src.scripts.pgs_pdlm
