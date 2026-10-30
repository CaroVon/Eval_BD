#!/bin/bash
export KLAB=/net/spaces/jevans/klab/caroyf HF_HOME=/net/spaces/jevans/klab/caroyf/data/ICML/hf_cache
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qc-tools
python - <<'EOF'
from huggingface_hub import HfApi
i = HfApi().model_info("Qwen/Qwen2.5-7B-Instruct")
print("gated:", i.gated, "license:", (i.card_data or {}).get("license"), "sha:", i.sha)
EOF
python ~/myproject/ICML/scripts/dl_model.py Qwen/Qwen2.5-7B-Instruct