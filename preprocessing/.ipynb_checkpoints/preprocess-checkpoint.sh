#!/bin/bash

################
# SBATCH OPTIONS
################

#SBATCH --job-name=neuroimage # job name for queue (optional)
#SBATCH --partition=low    # partition (optional, default=low) 
#SBATCH --error=preprocess.err     # file for stderr (optional)
#SBATCH --output=preprocess.out    # file for stdout (optional)
#SBATCH --time=3-24:00:00    # max runtime of job hours:minutes:seconds
#SBATCH --nodes=1          # use 1 node
#SBATCH --ntasks=1         # use 1 task
#SBATCH --cpus-per-task=1  # use 1 CPU core
#SBATCH --mail-user=khern045@berkeley.edu
#SBATCH --mail-type=ALL

###################
# Command(s) to run
###################

module load python

source /scratch/users/neuroimage/conda/venv/bin/activate

python -u preprocess.py

echo "Finished Program"
