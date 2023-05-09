#!/bin/bash

################
# SBATCH OPTIONS
################

#SBATCH --job-name=cneuroimage # job name for queue (optional)
#SBATCH --partition=low    # partition (optional, default=low) 
#SBATCH --error=clinical.err     # file for stderr (optional)
#SBATCH --output=clinical.out    # file for stdout (optional)
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

jupyter nbconvert --to pdf --execute clinical.ipynb --output clinical.pdf

echo "Finished Program"
