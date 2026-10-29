"""Download a parent model at a pinned commit into data/ICML/models/hf/<name>."""
import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

repo = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-3B-Instruct"
sha = HfApi().model_info(repo).sha
dest = Path(os.environ["KLAB"]) / "data" / "ICML" / "models" / "hf" / repo.split("/")[-1]

snapshot_download(repo, revision=sha, local_dir=dest,
                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "LICENSE*"])
(dest / "PINNED_REVISION.json").write_text(json.dumps({"repo": repo, "revision": sha}, indent=1))
print("MODEL_DONE", repo, sha, dest)
