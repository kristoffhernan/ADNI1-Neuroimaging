#!/bin/bash


################
# SBATCH OPTIONS
################

#SBATCH --job-name=utilsneuroimage # job name for queue (optional)
#SBATCH --partition=low    # partition (optional, default=low) 
#SBATCH --error=utils.err     # file for stderr (optional)
#SBATCH --out=utils.out     # file for out (optional)
#SBATCH --time=3-24:00:00    # max runtime of job hours:minutes:seconds
#SBATCH --nodes=1         # use 1 node
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1         # use 1 task
#SBATCH --cpus-per-task=1  # use 1 CPU core
#SBATCH --mail-user=khern045@berkeley.edu
#SBATCH --mail-type=ALL

###################
# Command(s) to run
###################

module load ../../R/4.0.2

Rscript -e "rmarkdown::render('utils.Rmd', output_format = 'pdf_document')"
