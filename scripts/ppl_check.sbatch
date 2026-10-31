#!/bin/bash
#SBATCH --job-name=ppl_check
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1,local:60G
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=/net/spaces/jevans/klab/caroyf/logs/ICML/%x_%j.out

# Perplexity check of the dequantized models (Track A validity): WikiText-2 test, n_ctx 512, for
# each version both the GGUF under llama.cpp (CUDA, sm_80 build) and the dequantized fp32
# safetensors under transformers with the same chunking (src/ppl_hf.py). The parent is compared as
# bf16.gguf vs the HF checkpoint. Each run is skipped when its result exists.
# Env: NAME (gguf / dequant prefix), HFDIR.
set -euo pipefail
K=/net/spaces/jevans/klab/caroyf
export HF_HOME=$K/data/ICML/hf_cache
LC=~/myproject/ICML/third_party/llama.cpp
G=$K/data/ICML/models/gguf/${NAME:?}
D=$K/data/ICML/models/dequant_fp32/$NAME
HF=$K/data/ICML/models/hf/${HFDIR:?}
TXT=$K/data/ICML/datasets/wikitext2/test.raw
OUT=$K/outputs/ICML/ppl/$NAME
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,driver_version,compute_cap,uuid --format=csv | tee -a "$OUT/gpu.csv"
set +u; source ~/myproject/ICML/scripts/stage_env.sh qc-vllm; set -u
cd ~/myproject/ICML/src

for V in parent Q8_0 Q6_K Q4_K_M Q3_K_M; do
  if [ "$V" = parent ]; then GG=$G/bf16.gguf; MD=$HF; else GG=$G/$V.gguf; MD=$D-$V; fi
  if ! grep -q "Final estimate" "$OUT/llamacpp_$V.log" 2>/dev/null; then
    LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH:-} \
      "$LC/build/bin/llama-perplexity" -m "$GG" -f "$TXT" -c 512 -b 2048 -ngl 99 --seed 0 \
      > "$OUT/llamacpp_$V.log" 2>&1
  fi
  grep "Final estimate" "$OUT/llamacpp_$V.log"
  [ -s "$OUT/hf_$V.json" ] || python ppl_hf.py --model "$MD" --text "$TXT" --out "$OUT/hf_$V.json"
done
echo PPL_CHECK_DONE
