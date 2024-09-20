#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH time=12000:00
python train.py \
  --model vanilla_bert \
  --datafiles ../../mqc/data/wt/queries.tsv ../../mqc/data/docs.tsv \
  --qrels ../../mqc/data/random/qrels \
  --train_pairs ../../mqc/data/random/train1.pairs \
  --valid_run ../../mqc/data/random/valid1.run \
  --model_out_dir models/vbert \
  --imgdict ../../mqc/data/random/img_pairs \