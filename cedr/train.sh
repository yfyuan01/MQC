#!/bin/bash
#SBATCH --job-name=bert-train
#SBATCH --cpus-per-task=4 --mem=32G
#SBATCH -p gpu --gres=gpu:1
#SBATCH --output=slurm_logs/%x-%j.out
python cedr/train.py --model vanilla_bert  --datafiles data/clari/facets.tsv data/clari/documents.tsv --qrels data/clari/qrels --train_pairs data/clari/train.humananswers.pairs --valid_run data/clari/val.humananswers.run --model_out_dir models/vbhuman
