#!/bin/bash
# Download Qwen2.5-14B-Instruct (pinned revision). Resumable: re-running skips finished files.
export KLAB=/net/spaces/jevans/klab/caroyf HF_HOME=/net/spaces/jevans/klab/caroyf/data/ICML/hf_cache
export HF_TOKEN_PATH=$HOME/.config/huggingface/token
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qc-tools
python - <<'EOF'
from huggingface_hub import HfApi
i = HfApi().model_info("Qwen/Qwen2.5-14B-Instruct")
print("gated:", i.gated, "license:", (i.card_data or {}).get("license"), "sha:", i.sha)
EOF
python ~/myproject/ICML/scripts/dl_model.py Qwen/Qwen2.5-14B-Instruct
