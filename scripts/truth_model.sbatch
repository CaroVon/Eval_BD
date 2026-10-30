#!/bin/bash
#SBATCH --job-name=truth_model
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1,local:80G
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/net/spaces/jevans/klab/caroyf/logs/ICML/%x_%j.out

# Truth matrix + margin profiles for any parent on any dataset subset, in a mode whose
# four-layer determinism check has already passed for this parent (see DECISIONS.md D10).
# Stages are skipped when their output exists (resume after preemption).
# Env: HFDIR, NAME (dequant prefix), DATASET, IDSFILE, TAG, EXTRA (e.g. --enforce-eager).
set -euo pipefail
K=/net/spaces/jevans/klab/caroyf
export HF_HOME=$K/data/ICML/hf_cache
export VLLM_BATCH_INVARIANT=1
EXTRA=${EXTRA:-}
PARENT=$K/data/ICML/models/hf/${HFDIR:?}
CH=$K/data/ICML/models/dequant_fp32/${NAME:?}
DATA=$K/data/ICML/datasets/${DATASET:?}
IDS=$DATA/${IDSFILE:?}
OUT=$K/outputs/ICML/truth/${TAG:?}_${SLURM_JOB_ID}
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,driver_version,compute_cap,uuid --format=csv | tee -a "$OUT/gpu.csv"
echo "parent=$HFDIR dataset=$DATASET ids=$IDSFILE extra=$EXTRA restart=${SLURM_RESTART_COUNT:-0}" | tee -a "$OUT/job.txt"
set +u; source ~/myproject/ICML/scripts/stage_env.sh qc-vllm; set -u
cd ~/myproject/ICML/src

if [ ! -s "$OUT/parent_gen/gen_A.jsonl" ]; then
  python det_check.py --model $PARENT --data-dir $DATA --ids $IDS --out-dir "$OUT/parent_gen" \
    --dtype float32 --phases gen --tag A $EXTRA
fi
S="--parent-gen $OUT/parent_gen/gen_A.jsonl --data-dir $DATA --ids $IDS --out-dir $OUT/screen --dtype float32 --topk 2 $EXTRA"
for M in parent Q8_0 Q6_K Q4_K_M Q3_K_M; do
  [ -s "$OUT/screen/$M/summary.json" ] && continue
  if [ "$M" = parent ]; then MP=$PARENT; else MP=$CH-$M; fi
  python child_screen.py $S --name $M --model $MP
done
for M in parent Q8_0 Q6_K Q4_K_M Q3_K_M; do
  [ -s "$OUT/margins/margins_$M.npz" ] && continue
  if [ "$M" = parent ]; then MP=$PARENT; else MP=$CH-$M; fi
  python margin_profile.py --model $MP --name $M --parent-gen $OUT/parent_gen/gen_A.jsonl \
    --data-dir $DATA --ids $IDS --out-dir $OUT/margins $EXTRA
done
echo TRUTH_MODEL_DONE
