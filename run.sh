#!/bin/bash
#SBATCH -J psn_train
#SBATCH -A r01682
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=%x.%j.out

module load conda
conda activate psn

python train.py

python decoder_mod.py

python imputation.py

python classification.py
