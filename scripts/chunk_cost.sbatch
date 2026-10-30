#!/bin/bash
#SBATCH --job-name=chunk_cost
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1,local:60G
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=/net/spaces/jevans/klab/caroyf/logs/ICML/%x_%j.out

# Cost only (prefix caching on): per-token cost of verifying L-token chunks vs decode.
set -euo pipefail
K=/net/spaces/jevans/klab/caroyf
export HF_HOME=$K/data/ICML/hf_cache
export VLLM_BATCH_INVARIANT=1
DATA=$K/data/ICML/datasets/gsm8k
OUT=$K/outputs/ICML/cost/chunk_cost_fp32_Q8_0_${SLURM_JOB_ID}.json
[ -s "$OUT" ] && exit 0
nvidia-smi --query-gpu=name,driver_version,uuid --format=csv
set +u; source ~/myproject/ICML/scripts/stage_env.sh qc-vllm; set -u
cd ~/myproject/ICML/src
python chunk_cost.py --model $K/data/ICML/models/dequant_fp32/qwen2.5-3b-instruct-Q8_0 \
  --parent-gen $K/outputs/ICML/truth/gsm8k_full_fp32_qwen2.5-3b_2273994/parent_gen/gen_A.jsonl \
  --data-dir $DATA --ids $DATA/all_test_ids.json --out "$OUT"
