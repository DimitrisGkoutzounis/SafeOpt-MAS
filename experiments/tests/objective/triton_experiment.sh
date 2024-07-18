#!/bin/bash
#SBATCH --job-name=experiment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:30:00



source activate conda-example  

srun python3 experiment3_objective_totalcost.py

