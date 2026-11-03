#!/bin/bash
#SBATCH --job-name=speed_bs
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1,local:80G
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/net/spaces/jevans/klab/caroyf/logs/ICML/%x_%j.out

# End-to-end wall-clock at a given decode batch (src/speed_g1.py), fp32, GSM8K 1,319.
# Per version: full generation, screening + continuation without and with prefix caching.
# Finished variants are skipped on restart.
# Env: NAME (dequant prefix), TRUTH (truth dir name), MAXSEQS (default 32), VERSIONS (default
#      "Q8_0 Q6_K"), EXTRA (e.g. --enforce-eager, as in the truth run), TAG.
set -euo pipefail
K=/net/spaces/jevans/klab/caroyf
export HF_HOME=$K/data/ICML/hf_cache
export VLLM_BATCH_INVARIANT=1
MAXSEQS=${MAXSEQS:-32}
VERSIONS=${VERSIONS:-Q8_0 Q6_K}
EXTRA=${EXTRA:-}
CH=$K/data/ICML/models/dequant_fp32/${NAME:?}
T=$K/outputs/ICML/truth/${TRUTH:?}
DATA=$K/data/ICML/datasets/gsm8k
IDS=$DATA/all_test_ids.json
OUT=$K/outputs/ICML/speed_bs/${TAG:?}_bs${MAXSEQS}
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,driver_version,compute_cap,uuid --format=csv | tee -a "$OUT/gpu.csv"
echo "job=$SLURM_JOB_ID node=$(hostname) name=$NAME maxseqs=$MAXSEQS extra=$EXTRA restart=${SLURM_RESTART_COUNT:-0}" | tee -a "$OUT/job.txt"
set +u; source ~/myproject/ICML/scripts/stage_env.sh qc-vllm; set -u
cd ~/myproject/ICML/src
C="--data-dir $DATA --ids $IDS --out-dir $OUT --max-num-seqs $MAXSEQS --dtype float32 $EXTRA"
run() { local name=$1; shift; [ -s "$OUT/$name.json" ] && return 0; python speed_g1.py --name $name "$@" $C; }
for V in $VERSIONS; do
  R="--parent-gen $T/parent_gen/gen_A.jsonl --ref $T/screen/$V/records.jsonl"
  run gen_$V --mode gen --model $CH-$V
  run scr_${V}_k1 --mode screen --skip-full --model $CH-$V --topk 1 $R
  run scr_${V}_k1_pc --mode screen --skip-full --model $CH-$V --topk 1 --prefix-caching $R
done
echo SPEED_BS_DONE
