#!/bin/bash
#SBATCH --job-name=bert-test
#SBATCH --cpus-per-task=4 --mem=32G
#SBATCH -p gpu --gres=gpu:1
#SBATCH --output=slurm_logs/%x-%j.out
python cedr/rerank.py --model vanilla_bert --datafiles data/clari/facets.tsv data/clari/documents.tsv --run data/clari/test.llmanswers.run --model_weights models/vbllm/weights.p --out_path models/vbllm/test.run
