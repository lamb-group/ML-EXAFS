#!/bin/bash
#SBATCH --job-name=ONNE_Model
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=3:00:00
#SBATCH --gpus=1
#SBATCH --constraint=v100
#SBATCH --output=md_model_es_%j.out
#SBATCH --error=md_model_es_%j.err
#SBATCH --partition=gpu

export OMP_NUM_THREADS=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml-exafs

module load cuda/12.8

echo "Total CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Nodes allocated: $SLURM_JOB_NUM_NODES"

time python model_tuning.py
