#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --time=12000:00
bash scripts/GenerativeRetrieval_VLT5.sh 1