#!/bin/bash
#SBATCH --job-name=ONNE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=data_gen_%j.out
#SBATCH --error=data_gen_%j.err
#SBATCH --partition=cpu

export OMP_NUM_THREADS=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml-exafs

echo "Total CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Nodes allocated: $SLURM_JOB_NUM_NODES"
echo "Input:"
echo "---------------------------"
cat onne_main_V2.py
echo "---------------------------"

time python onne_main.py --nprocs $SLURM_CPUS_PER_TASK
