#!/bin/bash
#SBATCH --job-name=speed_fewshot
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1,local:80G
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/net/spaces/jevans/klab/caroyf/logs/ICML/%x_%j.out

# Natural memory limit without a batch cap: GSM8K 8-shot (datasets/gsm8k_8shot), max_num_seqs 256,
# fp32. Prefix caching is on for every variant (the 8-shot prefix is shared, so a fair engine caches
# it); one full generation without caching is kept as a reference for concurrency and as a check that
# caching does not change outputs. Parent and child references are generated here (no truth matrix).
# Finished variants are skipped on restart.
# Env: HFDIR, NAME (dequant prefix), TAG, EXTRA (e.g. --enforce-eager), VERSIONS (default "Q8_0 Q6_K").
set -euo pipefail
K=/net/spaces/jevans/klab/caroyf
export HF_HOME=$K/data/ICML/hf_cache
export VLLM_BATCH_INVARIANT=1
VERSIONS=${VERSIONS:-Q8_0 Q6_K}
EXTRA=${EXTRA:-}
PARENT=$K/data/ICML/models/hf/${HFDIR:?}
CH=$K/data/ICML/models/dequant_fp32/${NAME:?}
DATA=$K/data/ICML/datasets/gsm8k_8shot
IDS=$DATA/all_test_ids.json
OUT=$K/outputs/ICML/speed_fewshot/${TAG:?}
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,driver_version,compute_cap,uuid --format=csv | tee -a "$OUT/gpu.csv"
echo "job=$SLURM_JOB_ID node=$(hostname) parent=$HFDIR name=$NAME extra=$EXTRA restart=${SLURM_RESTART_COUNT:-0}" | tee -a "$OUT/job.txt"
set +u; source ~/myproject/ICML/scripts/stage_env.sh qc-vllm; set -u
cd ~/myproject/ICML/src
C="--data-dir $DATA --ids $IDS --out-dir $OUT --max-num-seqs 256 --dtype float32 $EXTRA"
run() { local name=$1; shift; [ -s "$OUT/$name.json" ] && return 0; python speed_g1.py --name $name "$@" $C; }
run gen_parent_pc --mode gen --model $PARENT --prefix-caching
for V in $VERSIONS; do
  run gen_${V}_pc --mode gen --model $CH-$V --prefix-caching
  run scr_${V}_pc --mode screen --skip-full --model $CH-$V --topk 1 --prefix-caching \
    --parent-gen $OUT/gen_parent_pc_gen.jsonl --ref $OUT/gen_${V}_pc_gen.jsonl
done
V0=$(echo $VERSIONS | cut -d' ' -f1)
run gen_${V0}_nopc --mode gen --model $CH-$V0
echo SPEED_FEWSHOT_DONE
