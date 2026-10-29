#!/bin/bash
#SBATCH --job-name=envcheck
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100
#SBATCH --time=00:10:00
#SBATCH --output=/net/spaces/jevans/klab/caroyf/logs/ICML/%x_%j.out
hostname
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,uuid --format=csv
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qc-vllm
python - <<'EOF'
import torch, vllm
print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    print("matmul ok:", (x @ x).float().abs().mean().item() > 0)
EOF