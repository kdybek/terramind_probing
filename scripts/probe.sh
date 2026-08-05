#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --array=0-9
#SBATCH --account=plgcredibleai2026-cpu
#SBATCH --partition=plgrid
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

ml Python/3.11.5

export XDG_CACHE_HOME=$SCRATCH/.cache

mkdir -p $SCRATCH/terramind_probing
cd $SCRATCH/terramind_probing
cp -rf ~/terramind_probing/* .

source .venv_cpu/bin/activate

python probe.py $SLURM_ARRAY_TASK_ID
