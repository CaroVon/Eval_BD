"""Download the WikiText-2 (raw) test split at a pinned revision and write it as one text file.

Rows of the HF parquet are the lines of wiki.test.raw (each ending in a newline), so joining them
reproduces the file llama.cpp's perplexity tool is usually run on.
Output: data/ICML/datasets/wikitext2/{test.raw, meta.json}
"""
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

REPO = "Salesforce/wikitext"
FILE = "wikitext-2-raw-v1/test-00000-of-00001.parquet"
sha = HfApi().dataset_info(REPO).sha
path = hf_hub_download(REPO, FILE, repo_type="dataset", revision=sha)
lines = pq.read_table(path).column("text").to_pylist()
out = Path(os.environ["KLAB"]) / "data" / "ICML" / "datasets" / "wikitext2"
out.mkdir(parents=True, exist_ok=True)
text = "".join(lines)
(out / "test.raw").write_text(text, encoding="utf-8")
(out / "meta.json").write_text(json.dumps({"repo": REPO, "file": FILE, "revision": sha, "rows": len(lines),
                                           "chars": len(text)}, indent=1))
print("WIKITEXT_DONE", sha, len(lines), len(text))
