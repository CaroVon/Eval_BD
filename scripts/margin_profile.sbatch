#!/bin/bash
#SBATCH --job-name=margins
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1,local:60G
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/net/spaces/jevans/klab/caroyf/logs/ICML/%x_%j.out

# Analysis 1: margin profiles along the parent outputs, for parent + 4 children, on the
# GSM8K full and MMLU-Pro pilot truth matrices (same fp32 config as the truth matrices).
set -euo pipefail
K=/net/spaces/jevans/klab/caroyf
export HF_HOME=$K/data/ICML/hf_cache
export VLLM_BATCH_INVARIANT=1
set +u; source ~/myproject/ICML/scripts/stage_env.sh qc-vllm; set -u
cd ~/myproject/ICML/src
PARENT=$K/data/ICML/models/hf/Qwen2.5-3B-Instruct

run_set() {  # truth_dir dataset ids_file
  local T=$1 DATA=$K/data/ICML/datasets/$2 IDS=$K/data/ICML/datasets/$2/$3
  for M in parent Q8_0 Q6_K Q4_K_M Q3_K_M; do
    [ -s "$T/margins/margins_$M.npz" ] && continue
    if [ "$M" = parent ]; then MP=$PARENT; else MP=$K/data/ICML/models/dequant_fp32/qwen2.5-3b-instruct-$M; fi
    python margin_profile.py --model $MP --name $M --parent-gen $T/parent_gen/gen_A.jsonl \
      --data-dir $DATA --ids $IDS --out-dir $T/margins
  done
}
run_set $K/outputs/ICML/truth/gsm8k_full_fp32_qwen2.5-3b_2273994 gsm8k all_test_ids.json
run_set $K/outputs/ICML/truth/mmlupro_pilot1000_fp32_qwen2.5-3b_2274801 mmlu_pro pilot_1000_ids.json
echo MARGINS_DONE
