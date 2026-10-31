#!/bin/bash
#SBATCH --job-name=trackb_cont
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1,local:60G
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/net/spaces/jevans/klab/caroyf/logs/ICML/%x_%j.out

# Track B continuation check (src/trackb_cont.py) against existing Track B greedy outputs.
# Env: NAME, HFDIR, VERSIONS, DATASET, IDSFILE, TAG (dir with gen_<V>.jsonl from trackb.sbatch).
set -euo pipefail
K=/net/spaces/jevans/klab/caroyf
export HF_HOME=$K/data/ICML/hf_cache
LC=~/myproject/ICML/third_party/llama.cpp
G=$K/data/ICML/models/gguf/${NAME:?}
HF=$K/data/ICML/models/hf/${HFDIR:?}
DATA=$K/data/ICML/datasets/${DATASET:-gsm8k}
IDS=$DATA/${IDSFILE:-determinism_subset_ids.json}
OUT=$K/outputs/ICML/trackb/${TAG:?}
set +u; source ~/myproject/ICML/scripts/stage_env.sh qc-vllm; set -u
cd ~/myproject/ICML/src
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH:-}
PORT=$((20000 + SLURM_JOB_ID % 20000))
for V in ${VERSIONS:?}; do
  "$LC/build/bin/llama-server" -m "$G/$V.gguf" -ngl 99 -c 4096 -np 1 --port $PORT --host 127.0.0.1 \
    --seed 0 > "$OUT/server_cont_$V.log" 2>&1 &
  SP=$!
  for i in $(seq 120); do curl -sf http://127.0.0.1:$PORT/health > /dev/null && break; sleep 2; done
  python trackb_cont.py --port $PORT --model-dir "$HF" --data-dir "$DATA" --ids "$IDS" \
    --ref "$OUT/gen_$V.jsonl" --out "$OUT/cont_$V.jsonl"
  kill $SP; wait $SP || true
done
echo TRACKB_CONT_DONE
