#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=1:00:00
#SBATCH --account=plgcredibleai2026-cpu
#SBATCH --partition=plgrid
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

ml Python/3.11.5

export XDG_CACHE_HOME=$SCRATCH/.cache

mkdir -p $SCRATCH/terramind_probing
cd $SCRATCH/terramind_probing
cp -rf ~/terramind_probing/* .

source .venv_cpu/bin/activate

python collate_res.py
