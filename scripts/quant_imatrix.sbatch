#!/bin/bash
#SBATCH --job-name=quant_imatrix
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1,local:60G
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=/net/spaces/jevans/klab/caroyf/logs/ICML/%x_%j.out

# imatrix ladder for one parent (env NAME, HFDIR): importance matrix of the existing bf16 GGUF on
# WikiText-2 train (CHUNKS x 512 tokens), then {Q6_K,Q4_K_M,Q3_K_M} quantized with it into
# models/gguf/$NAME-imat, dequantized to fp32 as models/dequant_fp32/$NAME-imat-<Q>.
# Q8_0 is skipped (llama-quantize ignores the imatrix for Q8_0). Every step is skipped if its
# output exists, so a requeued job resumes.
set -euo pipefail
K=/net/spaces/jevans/klab/caroyf
NAME=${NAME:?}
HF=$K/data/ICML/models/hf/${HFDIR:?}
G0=$K/data/ICML/models/gguf/$NAME
G=$K/data/ICML/models/gguf/$NAME-imat
D=$K/data/ICML/models/dequant_fp32
CAL=$K/data/ICML/datasets/wikitext2/train.raw
CHUNKS=${CHUNKS:-200}
LC=~/myproject/ICML/third_party/llama.cpp
PIN=~/myproject/ICML/third_party/llama.cpp.PINNED
mkdir -p "$G"
set +u; source ~/myproject/ICML/scripts/stage_env.sh qc-vllm; set -u
EXTRA=$K/data/ICML/envs_pack/extra_site_qc-vllm
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH:-}
cat "$PIN" | tee "$G/llamacpp_pin.txt"

if [ ! -s "$G/imatrix.gguf" ]; then
  "$LC/build/bin/llama-imatrix" -m "$G0/bf16.gguf" -f "$CAL" -c 512 --chunks "$CHUNKS" -ngl 99 \
    -o "$G/imatrix.tmp.gguf" 2>&1 | tail -20
  mv "$G/imatrix.tmp.gguf" "$G/imatrix.gguf"
  echo "calibration=wikitext2/train.raw chunks=$CHUNKS ctx=512 model=$G0/bf16.gguf" > "$G/imatrix_meta.txt"
fi
for Q in Q6_K Q4_K_M Q3_K_M; do
  if [ ! -s "$G/$Q.gguf" ]; then
    "$LC/build/bin/llama-quantize" --imatrix "$G/imatrix.gguf" "$G0/bf16.gguf" "$G/$Q.gguf.tmp" "$Q" \
      "$SLURM_CPUS_PER_TASK"
    mv "$G/$Q.gguf.tmp" "$G/$Q.gguf"
  fi
  OUT=$D/$NAME-imat-$Q
  if [ ! -s "$OUT/MANIFEST.json" ]; then
    rm -rf "$OUT"
    PYTHONPATH=$EXTRA python ~/myproject/ICML/src/dequant_gguf.py --gguf "$G/$Q.gguf" --ref "$HF" \
      --out "$OUT" --quant "$Q" --llamacpp-pin "$PIN" --dtype float32
  fi
done
ls -la "$G"
echo IMATRIX_DONE
