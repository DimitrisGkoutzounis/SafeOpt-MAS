#!/bin/bash
#SBATCH --job-name=experiment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:30:00



source activate conda-example  # Activate your virtual environment

srun python3 experiment2_objective.py

