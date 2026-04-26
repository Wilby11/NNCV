#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=06:00:00

# Install albumentations inside the container (lightweight, no rebuild)
srun apptainer exec --nv --env-file .env container.sif pip3 install albumentations

srun apptainer exec --nv --env-file .env container.sif /bin/bash main.sh