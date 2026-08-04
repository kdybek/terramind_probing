#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --account=plgcredibleai2026-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

ml Python/3.11.5

export XDG_CACHE_HOME=$SCRATCH/.cache
export HF_TOKEN=(cat ~/.hf_token)

mkdir -p $SCRATCH/terramind_probing
cd $SCRATCH/terramind_probing
cp -rf ~/terramind_probing/* .

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

hf download ibm-esa-geospatial/TerraMesh --repo-type dataset --include "val/*" --local-dir data/TerraMesh
