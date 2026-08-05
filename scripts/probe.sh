#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --account=plgcredibleai2026-cpu
#SBATCH --partition=plgrid
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

ml Python/3.11.5

export XDG_CACHE_HOME=$SCRATCH/.cache
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p $SCRATCH/terramind_probing
cd $SCRATCH/terramind_probing
cp -rf ~/terramind_probing/* .

source .venv_cpu/bin/activate

python -u probe.py
