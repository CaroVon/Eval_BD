#!/bin/bash
#SBATCH --job-name=gsm8k_full
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1,local:60G
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=/net/spaces/jevans/klab/caroyf/logs/ICML/%x_%j.out

# Truth matrix on the full GSM8K test set (1,319) in the adopted fp32 config:
# parent generation, then exact screening of parent (control) + 4 children.
# Each stage is skipped if its output exists, so a preempted+requeued job resumes.
set -euo pipefail
K=/net/spaces/jevans/klab/caroyf
export HF_HOME=$K/data/ICML/hf_cache
export VLLM_BATCH_INVARIANT=1
DATA=$K/data/ICML/datasets/gsm8k
IDS=$DATA/all_test_ids.json
OUT=$K/outputs/ICML/truth/gsm8k_full_fp32_qwen2.5-3b_${SLURM_JOB_ID}
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,driver_version,compute_cap,uuid --format=csv | tee -a "$OUT/gpu.csv"
echo "restart_count=${SLURM_RESTART_COUNT:-0}"

set +u; source ~/myproject/ICML/scripts/stage_env.sh qc-vllm; set -u
cd ~/myproject/ICML/src
PARENT=$K/data/ICML/models/hf/Qwen2.5-3B-Instruct

if [ ! -s "$OUT/parent_gen/gen_A.jsonl" ]; then
  python det_check.py --model $PARENT --data-dir $DATA --ids $IDS \
    --out-dir "$OUT/parent_gen" --dtype float32 --phases gen --tag A
fi

S="--parent-gen $OUT/parent_gen/gen_A.jsonl --data-dir $DATA --ids $IDS --out-dir $OUT/screen --dtype float32 --topk 2"
for M in parent Q8_0 Q6_K Q4_K_M Q3_K_M; do
  [ -s "$OUT/screen/$M/summary.json" ] && continue
  if [ "$M" = parent ]; then MP=$PARENT; else MP=$K/data/ICML/models/dequant_fp32/qwen2.5-3b-instruct-$M; fi
  python child_screen.py $S --name $M --model $MP
done
echo GSM8K_FULL_DONE
