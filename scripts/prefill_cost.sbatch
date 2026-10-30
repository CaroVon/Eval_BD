#!/bin/bash
#SBATCH --job-name=prefill_cost
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1,local:60G
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --output=/net/spaces/jevans/klab/caroyf/logs/ICML/%x_%j.out

# Prefill vs decode cost for the fp32 Q8_0 child. GPU type can be overridden on the sbatch
# command line (--gres/--constraint); this is a cost measurement only, not truth-matrix data.
# Env: IDS (default: full GSM8K test), PARENT_GEN (default: full-GSM8K fp32 parent outputs).
set -euo pipefail
K=/net/spaces/jevans/klab/caroyf
export HF_HOME=$K/data/ICML/hf_cache
export VLLM_BATCH_INVARIANT=1
DATA=$K/data/ICML/datasets/gsm8k
IDS=${IDS:-$DATA/all_test_ids.json}
PARENT_GEN=${PARENT_GEN:-$K/outputs/ICML/truth/gsm8k_full_fp32_qwen2.5-3b_2273994/parent_gen/gen_A.jsonl}
MODEL=${MODEL:-$K/data/ICML/models/dequant_fp32/qwen2.5-3b-instruct-Q8_0}
MTAG=${MTAG:-fp32_Q8_0}
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
nvidia-smi --query-gpu=name,driver_version,compute_cap,uuid --format=csv
set +u; source ~/myproject/ICML/scripts/stage_env.sh qc-vllm; set -u
cd ~/myproject/ICML/src
python prefill_cost.py --model "$MODEL" \
  --parent-gen "$PARENT_GEN" --data-dir $DATA --ids "$IDS" \
  --out $K/outputs/ICML/cost/prefill_cost_${MTAG}_n$(python -c "import json;print(len(json.load(open('$IDS'))))")_${GPU}_${SLURM_JOB_ID}.json
