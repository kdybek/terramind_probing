#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --account=plgcredibleai2026-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

ml Python/3.11.5

export XDG_CACHE_HOME=$SCRATCH/.cache

mkdir -p $SCRATCH/terramind_probing
cd $SCRATCH/terramind_probing
cp -rf ~/terramind_probing/* .

source .venv/bin/activate

python collect_latents.py
