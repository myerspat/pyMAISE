#!/bin/bash

#SBATCH --job-name=ners551_project
#SBATCH --mail-user=myerspat@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2GB
#SBATCH --time=10:00:00
#SBATCH --account=engin1
#SBATCH --partition=standard

# Loads
module load python3.9-anaconda

python3 data_generation.py
python3 analysis.py
