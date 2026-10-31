"""Download a WikiText-2 (raw) split at a pinned revision and write it as one text file.

Rows of the HF parquet are the lines of wiki.<split>.raw (each ending in a newline), so joining them
reproduces the file llama.cpp's perplexity tool is usually run on.
Env SPLIT (default test; train is the imatrix calibration text), REVISION (default: latest, as used
for test.raw; the train split is pinned to the same revision b08601e).
Output: data/ICML/datasets/wikitext2/{<split>.raw, meta.json (test) or meta_<split>.json}
"""
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

REPO = "Salesforce/wikitext"
SPLIT = os.environ.get("SPLIT", "test")
FILE = f"wikitext-2-raw-v1/{SPLIT}-00000-of-00001.parquet"
sha = os.environ.get("REVISION") or HfApi().dataset_info(REPO).sha
path = hf_hub_download(REPO, FILE, repo_type="dataset", revision=sha)
lines = pq.read_table(path).column("text").to_pylist()
out = Path(os.environ["KLAB"]) / "data" / "ICML" / "datasets" / "wikitext2"
out.mkdir(parents=True, exist_ok=True)
text = "".join(lines)
(out / f"{SPLIT}.raw").write_text(text, encoding="utf-8")
meta = "meta.json" if SPLIT == "test" else f"meta_{SPLIT}.json"
(out / meta).write_text(json.dumps({"repo": REPO, "file": FILE, "revision": sha, "rows": len(lines),
                                    "chars": len(text)}, indent=1))
print("WIKITEXT_DONE", SPLIT, sha, len(lines), len(text))
