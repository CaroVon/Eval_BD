#!/bin/bash
#SBATCH --job-name=parent_suite
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1,local:80G
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/net/spaces/jevans/klab/caroyf/logs/ICML/%x_%j.out

# Full fp32 pipeline for one (additional) parent on GSM8K:
#   1. four-layer determinism check on the 200-question subset (stops the job if it fails)
#   2. truth matrix on all 1,319 questions: parent generation + screening of parent + 4 children
#   3. margin profiles along the parent outputs
# Every stage is skipped when its output exists, so a preempted+requeued job resumes.
# Env: HFDIR (dir under models/hf), NAME (gguf/dequant prefix), TAG (output name),
#      EXTRA (extra flags for every engine run, e.g. --enforce-eager).
set -euo pipefail
K=/net/spaces/jevans/klab/caroyf
export HF_HOME=$K/data/ICML/hf_cache
export VLLM_BATCH_INVARIANT=1
EXTRA=${EXTRA:-}
PARENT=$K/data/ICML/models/hf/${HFDIR:?}
CH=$K/data/ICML/models/dequant_fp32/${NAME:?}
DATA=$K/data/ICML/datasets/gsm8k
DET=$K/outputs/ICML/detcheck/${TAG:?}_fp32_${SLURM_JOB_ID}
OUT=$K/outputs/ICML/truth/gsm8k_full_fp32_${TAG}_${SLURM_JOB_ID}
mkdir -p "$DET" "$OUT"
nvidia-smi --query-gpu=name,driver_version,compute_cap,uuid --format=csv | tee -a "$OUT/gpu.csv"
echo "restart_count=${SLURM_RESTART_COUNT:-0} extra=$EXTRA" | tee -a "$OUT/job.txt"
set +u; source ~/myproject/ICML/scripts/stage_env.sh qc-vllm; set -u
cd ~/myproject/ICML/src

# 1. determinism
if [ ! -s "$DET/report.json" ]; then
  C="--model $PARENT --data-dir $DATA --ids $DATA/determinism_subset_ids.json --out-dir $DET --dtype float32 $EXTRA"
  python det_check.py $C --phases gen --tag A
  python det_check.py $C --phases gen,prefill,cont --tag B --ref A
  python det_check.py $C --phases gen --tag C --max-num-seqs 8 --shuffle 7
  python det_report.py "$DET" "$DATA"
fi
grep -q '"ALL_FOUR_LAYERS_PASS": true' "$DET/report.json" || { echo "DETERMINISM_FAILED"; exit 1; }

# 2. truth matrix
if [ ! -s "$OUT/parent_gen/gen_A.jsonl" ]; then
  python det_check.py --model $PARENT --data-dir $DATA --ids $DATA/all_test_ids.json \
    --out-dir "$OUT/parent_gen" --dtype float32 --phases gen --tag A $EXTRA
fi
S="--parent-gen $OUT/parent_gen/gen_A.jsonl --data-dir $DATA --ids $DATA/all_test_ids.json --out-dir $OUT/screen --dtype float32 --topk 2 $EXTRA"
for M in parent Q8_0 Q6_K Q4_K_M Q3_K_M; do
  [ -s "$OUT/screen/$M/summary.json" ] && continue
  if [ "$M" = parent ]; then MP=$PARENT; else MP=$CH-$M; fi
  python child_screen.py $S --name $M --model $MP
done

# 3. margins
for M in parent Q8_0 Q6_K Q4_K_M Q3_K_M; do
  [ -s "$OUT/margins/margins_$M.npz" ] && continue
  if [ "$M" = parent ]; then MP=$PARENT; else MP=$CH-$M; fi
  python margin_profile.py --model $MP --name $M --parent-gen $OUT/parent_gen/gen_A.jsonl \
    --data-dir $DATA --ids $DATA/all_test_ids.json --out-dir $OUT/margins $EXTRA
done
echo PARENT_SUITE_DONE
