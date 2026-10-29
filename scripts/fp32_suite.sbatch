#!/bin/bash
#SBATCH --job-name=fp32_suite
#SBATCH --partition=general
#SBATCH --qos=protected
#SBATCH --gres=gpu:a100:1,local:60G
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/net/spaces/jevans/klab/caroyf/logs/ICML/%x_%j.out

# fp32 execution config: four-layer determinism check on the parent, then exact
# screening of the fp32-dequantized children along the fp32 parent outputs.
set -euo pipefail
K=/net/spaces/jevans/klab/caroyf
export HF_HOME=$K/data/ICML/hf_cache
export VLLM_BATCH_INVARIANT=1
PARENT=$K/data/ICML/models/hf/Qwen2.5-3B-Instruct
DATA=$K/data/ICML/datasets/gsm8k
IDS=$DATA/determinism_subset_ids.json
DET=$K/outputs/ICML/detcheck/qwen2.5-3b_fp32_${SLURM_JOB_ID}
SCR=$K/outputs/ICML/pilot_screen/qwen2.5-3b_fp32_${SLURM_JOB_ID}
mkdir -p "$DET" "$SCR"
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,uuid --format=csv | tee "$DET/gpu.csv"

set +u; source ~/myproject/ICML/scripts/stage_env.sh qc-vllm; set -u
cd ~/myproject/ICML/src

# 1. determinism check in fp32
C="--model $PARENT --data-dir $DATA --ids $IDS --out-dir $DET --dtype float32"
python det_check.py $C --phases gen --tag A
python det_check.py $C --phases gen,prefill,cont --tag B --ref A
python det_check.py $C --phases gen --tag C --max-num-seqs 8 --shuffle 7
python det_report.py "$DET" "$DATA"
grep -q '"ALL_FOUR_LAYERS_PASS": true' "$DET/report.json" || { echo "FP32_DETERMINISM_FAILED"; exit 1; }

# 2. exact screening in fp32
S="--parent-gen $DET/gen_A.jsonl --data-dir $DATA --ids $IDS --out-dir $SCR --dtype float32"
python child_screen.py $S --name parent --model $PARENT
for Q in Q8_0 Q6_K Q4_K_M Q3_K_M; do
  python child_screen.py $S --name $Q --model $K/data/ICML/models/dequant_fp32/qwen2.5-3b-instruct-$Q
done
echo FP32_SUITE_DONE
