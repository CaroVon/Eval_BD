#!/bin/bash
#SBATCH --job-name=trackb
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1,local:60G
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=/net/spaces/jevans/klab/caroyf/logs/ICML/%x_%j.out

# Track B pilot: greedy outputs of the deployed GGUF under llama.cpp (llama-server, one slot, CUDA),
# with the Track A prompt ids, for comparison with the Track A truth matrix (src/trackb_gen.py).
# Output per version is appended question by question; a requeued job resumes.
# Env: NAME (gguf dir), HFDIR, VERSIONS (space-separated, may include bf16), DATASET, IDSFILE, TAG,
#      SERVER_EXTRA (extra llama-server flags, e.g. "-b 1 -ub 1").
set -euo pipefail
K=/net/spaces/jevans/klab/caroyf
export HF_HOME=$K/data/ICML/hf_cache
LC=~/myproject/ICML/third_party/llama.cpp
G=$K/data/ICML/models/gguf/${NAME:?}
HF=$K/data/ICML/models/hf/${HFDIR:?}
DATA=$K/data/ICML/datasets/${DATASET:-gsm8k}
IDS=$DATA/${IDSFILE:-all_test_ids.json}
OUT=$K/outputs/ICML/trackb/${TAG:?}
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,driver_version,compute_cap,uuid --format=csv | tee -a "$OUT/gpu.csv"
cat ~/myproject/ICML/third_party/llama.cpp.PINNED > "$OUT/llamacpp_pin.txt"
set +u; source ~/myproject/ICML/scripts/stage_env.sh qc-vllm; set -u
cd ~/myproject/ICML/src
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH:-}
PORT=$((20000 + SLURM_JOB_ID % 20000))

for V in ${VERSIONS:?}; do
  N=$(python -c "import json;print(len(json.load(open('$IDS'))))")
  if [ -s "$OUT/gen_$V.jsonl" ] && [ "$(wc -l < "$OUT/gen_$V.jsonl")" -ge "$N" ]; then continue; fi
  "$LC/build/bin/llama-server" -m "$G/$V.gguf" -ngl 99 -c 4096 -np 1 --port $PORT --host 127.0.0.1 \
    --seed 0 ${SERVER_EXTRA:-} > "$OUT/server_$V.log" 2>&1 &
  SP=$!
  for i in $(seq 120); do
    curl -sf http://127.0.0.1:$PORT/health > /dev/null && break
    sleep 2
  done
  python trackb_gen.py --port $PORT --model-dir "$HF" --data-dir "$DATA" --ids "$IDS" --out "$OUT/gen_$V.jsonl"
  kill $SP; wait $SP || true
done
echo TRACKB_DONE
