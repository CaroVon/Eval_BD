#!/bin/bash
#SBATCH --job-name=speed_g1
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1,local:80G
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/net/spaces/jevans/klab/caroyf/logs/ICML/%x_%j.out

# End-to-end wall-clock variants for the G1 speedup gate (src/speed_g1.py), Qwen2.5-3B GSM8K 1,319,
# compiled mode. Every variant runs in its own process; finished variants are skipped on restart.
# Env: OUT (output dir, default outputs/ICML/speed_g1), VERSIONS (default "Q8_0 Q6_K").
set -euo pipefail
K=/net/spaces/jevans/klab/caroyf
export HF_HOME=$K/data/ICML/hf_cache
export VLLM_BATCH_INVARIANT=1
VERSIONS=${VERSIONS:-Q8_0 Q6_K}
PARENT=$K/data/ICML/models/hf/Qwen2.5-3B-Instruct
CH=$K/data/ICML/models/dequant_fp32/qwen2.5-3b-instruct
DATA=$K/data/ICML/datasets/gsm8k
IDS=$DATA/all_test_ids.json
TRUTH=$K/outputs/ICML/truth/gsm8k_full_fp32_qwen2.5-3b_2273994
OUT=${OUT:-$K/outputs/ICML/speed_g1}
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,driver_version,compute_cap,uuid --format=csv | tee -a "$OUT/gpu.csv"
echo "job=$SLURM_JOB_ID node=$(hostname) restart=${SLURM_RESTART_COUNT:-0}" | tee -a "$OUT/job.txt"
set +u; source ~/myproject/ICML/scripts/stage_env.sh qc-vllm; set -u
cd ~/myproject/ICML/src
C="--data-dir $DATA --ids $IDS --out-dir $OUT"
run() { local name=$1; shift; [ -s "$OUT/$name.json" ] && return 0; python speed_g1.py --name $name "$@" $C; }

# fp32 (adopted configuration): parent outputs and child references from the truth matrix
for V in $VERSIONS; do
  run gen_fp32_$V --mode gen --model $CH-$V --dtype float32
  run scr_fp32_${V}_k2 --mode screen --skip-full --model $CH-$V --dtype float32 --topk 2 \
    --parent-gen $TRUTH/parent_gen/gen_A.jsonl --ref $TRUTH/screen/$V/records.jsonl
  run scr_fp32_${V}_k1 --mode screen --skip-full --model $CH-$V --dtype float32 --topk 1 \
    --parent-gen $TRUTH/parent_gen/gen_A.jsonl --ref $TRUTH/screen/$V/records.jsonl
  run scr_fp32_${V}_k1_pc --mode screen --skip-full --model $CH-$V --dtype float32 --topk 1 --prefix-caching \
    --parent-gen $TRUTH/parent_gen/gen_A.jsonl --ref $TRUTH/screen/$V/records.jsonl
done

# bf16: parent and child references generated here; topk 2 so exact ties are seen
run gen_bf16_parent --mode gen --model $PARENT --dtype bfloat16
for V in $VERSIONS; do
  run gen_bf16_$V --mode gen --model $CH-$V --dtype bfloat16
  run scr_bf16_${V}_k2 --mode screen --skip-full --model $CH-$V --dtype bfloat16 --topk 2 \
    --parent-gen $OUT/gen_bf16_parent_gen.jsonl --ref $OUT/gen_bf16_${V}_gen.jsonl
  run scr_bf16_${V}_k2_pc --mode screen --skip-full --model $CH-$V --dtype bfloat16 --topk 2 --prefix-caching \
    --parent-gen $OUT/gen_bf16_parent_gen.jsonl --ref $OUT/gen_bf16_${V}_gen.jsonl
done
echo SPEED_G1_DONE
