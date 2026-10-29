#!/bin/bash
#SBATCH --job-name=pilot_screen
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1,local:40G
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=/net/spaces/jevans/klab/caroyf/logs/ICML/%x_%j.out

# Pilot screening on the 200-question determinism subset: parent (control) + 4 children.
set -euo pipefail
K=/net/spaces/jevans/klab/caroyf
export HF_HOME=$K/data/ICML/hf_cache
export VLLM_BATCH_INVARIANT=1
DATA=$K/data/ICML/datasets/gsm8k
PARENT_GEN=$K/outputs/ICML/detcheck/qwen2.5-3b_bf16_2272378/gen_A.jsonl
OUT=$K/outputs/ICML/pilot_screen/qwen2.5-3b_${SLURM_JOB_ID}
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,driver_version,compute_cap,uuid --format=csv | tee "$OUT/gpu.csv"

set +u; source ~/myproject/ICML/scripts/stage_env.sh qc-vllm; set -u
cd ~/myproject/ICML/src
COMMON="--parent-gen $PARENT_GEN --data-dir $DATA --ids $DATA/determinism_subset_ids.json --out-dir $OUT"

python child_screen.py $COMMON --name parent --model $K/data/ICML/models/hf/Qwen2.5-3B-Instruct
for Q in Q8_0 Q6_K Q4_K_M Q3_K_M; do
  python child_screen.py $COMMON --name $Q --model $K/data/ICML/models/dequant_bf16/qwen2.5-3b-instruct-$Q
done
echo PILOT_DONE
