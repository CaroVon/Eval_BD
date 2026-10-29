#!/bin/bash
#SBATCH --job-name=detsmoke
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1,local:40G
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/net/spaces/jevans/klab/caroyf/logs/ICML/%x_%j.out

set -euo pipefail
K=/net/spaces/jevans/klab/caroyf
export HF_HOME=$K/data/ICML/hf_cache
export VLLM_BATCH_INVARIANT=1
MODEL=$K/data/ICML/models/hf/Qwen2.5-3B-Instruct
DATA=$K/data/ICML/datasets/gsm8k
OUT=$K/outputs/ICML/detcheck/smoke_${SLURM_JOB_ID}
mkdir -p "$OUT"
set +u; source ~/myproject/ICML/scripts/stage_env.sh qc-vllm; set -u
cd ~/myproject/ICML/src
COMMON="--model $MODEL --data-dir $DATA --ids $DATA/determinism_subset_ids.json --out-dir $OUT --limit 5 --max-tokens 128"
python det_check.py $COMMON --phases gen --tag A
python det_check.py $COMMON --phases gen,prefill,cont --tag B --ref A
python det_check.py $COMMON --phases gen --tag C --max-num-seqs 2 --shuffle 7
python det_report.py "$OUT" "$DATA"
