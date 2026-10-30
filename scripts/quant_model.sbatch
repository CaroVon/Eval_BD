#!/bin/bash
#SBATCH --job-name=quant_model
#SBATCH --partition=general
#SBATCH --gres=local:60G
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=/net/spaces/jevans/klab/caroyf/logs/ICML/%x_%j.out

# Track A ladder for one parent (env NAME, HFDIR, DTYPE): HF bf16 -> GGUF bf16 -> {Q8_0,Q6_K,Q4_K_M,Q3_K_M}
# -> dequantized safetensors in $DTYPE (bfloat16 default, or float32). No imatrix (pilot). Each step is skipped if its
# output already exists, so a requeued job resumes where it stopped.
set -euo pipefail
K=/net/spaces/jevans/klab/caroyf
NAME=${NAME:?}            # e.g. phi-3.5-mini-instruct
HF=$K/data/ICML/models/hf/${HFDIR:?}   # e.g. Phi-3.5-mini-instruct
G=$K/data/ICML/models/gguf/$NAME
DTYPE=${DTYPE:-bfloat16}
case $DTYPE in bfloat16) D=$K/data/ICML/models/dequant_bf16;; float32) D=$K/data/ICML/models/dequant_fp32;; esac
LC=~/myproject/ICML/third_party/llama.cpp
PIN=~/myproject/ICML/third_party/llama.cpp.PINNED
mkdir -p "$G" "$D"

set +u; source ~/myproject/ICML/scripts/stage_env.sh qc-vllm; set -u
# accelerate (needed by transformers GGUF loading) lives outside the packed env
EXTRA=$K/data/ICML/envs_pack/extra_site_qc-vllm
cat "$PIN"

if [ ! -s "$G/bf16.gguf" ]; then
  PYTHONPATH=$LC/gguf-py python "$LC/convert_hf_to_gguf.py" "$HF" --outtype bf16 --outfile "$G/bf16.gguf.tmp"
  mv "$G/bf16.gguf.tmp" "$G/bf16.gguf"
fi

for Q in Q8_0 Q6_K Q4_K_M Q3_K_M; do
  if [ ! -s "$G/$Q.gguf" ]; then
    "$LC/build/bin/llama-quantize" "$G/bf16.gguf" "$G/$Q.gguf.tmp" "$Q" "$SLURM_CPUS_PER_TASK"
    mv "$G/$Q.gguf.tmp" "$G/$Q.gguf"
  fi
  OUT=$D/$NAME-$Q
  if [ ! -s "$OUT/MANIFEST.json" ]; then
    rm -rf "$OUT"
    PYTHONPATH=$EXTRA python ~/myproject/ICML/src/dequant_gguf.py --gguf "$G/$Q.gguf" --ref "$HF" \
      --out "$OUT" --quant "$Q" --llamacpp-pin "$PIN" --dtype "$DTYPE"
  fi
done
ls -la "$G" "$D"
echo PIPELINE_DONE
