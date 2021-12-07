#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --nodelist=cs-gpu2
#SBATCH --output=jupyter.log


jupyter notebook --no-browser --port=8000

