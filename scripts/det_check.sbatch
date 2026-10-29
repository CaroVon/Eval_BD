#!/bin/bash
#SBATCH --job-name=detcheck
#SBATCH --partition=general
#SBATCH --qos=protected
#SBATCH --gres=gpu:a100:1,local:40G
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/net/spaces/jevans/klab/caroyf/logs/ICML/%x_%j.out

set -euo pipefail
K=/net/spaces/jevans/klab/caroyf
export HF_HOME=$K/data/ICML/hf_cache
export VLLM_BATCH_INVARIANT=1
MODEL=$K/data/ICML/models/hf/Qwen2.5-3B-Instruct
DATA=$K/data/ICML/datasets/gsm8k
OUT=$K/outputs/ICML/detcheck/qwen2.5-3b_bf16_${SLURM_JOB_ID}
mkdir -p "$OUT"

nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,uuid --format=csv | tee "$OUT/gpu.csv"
set +u; source ~/myproject/ICML/scripts/stage_env.sh qc-vllm; set -u
cd ~/myproject/ICML/src
COMMON="--model $MODEL --data-dir $DATA --ids $DATA/determinism_subset_ids.json --out-dir $OUT"

# Process 1: run A, repeat run B (same engine), prefill along A, continuation from A
python det_check.py $COMMON --phases gen --tag A
python det_check.py $COMMON --phases gen,prefill,cont --tag B --ref A
# Process 2: different max_num_seqs and shuffled submission order
python det_check.py $COMMON --phases gen --tag C --max-num-seqs 8 --shuffle 7

python det_report.py "$OUT" "$DATA"
