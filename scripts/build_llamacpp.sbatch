#!/bin/bash
#SBATCH --job-name=build_llamacpp
#SBATCH --partition=general
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/net/spaces/jevans/klab/caroyf/logs/ICML/%x_%j.out

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
set +u; conda activate qc-tools; set -u   # provides cmake + ninja
export PATH=/usr/local/cuda-13.1/bin:$PATH
cd ~/myproject/ICML/third_party/llama.cpp
cat ../llama.cpp.PINNED
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
  -DLLAMA_CURL=OFF
cmake --build build -j "$SLURM_CPUS_PER_TASK" \
  --target llama-quantize llama-perplexity llama-imatrix llama-server llama-cli
ls -la build/bin/
echo BUILD_DONE
