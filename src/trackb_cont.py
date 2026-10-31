"""Track B continuation check (layer L3 of the determinism check, for llama.cpp).

For each question with a reference greedy output y (from trackb_gen.py, same model), send
prompt + y[:t] for t = floor(n/3), floor(2n/3) and decode greedily; under prefill/decode consistency
the continuation equals y[t:]. Exact screening by prefill needs this to hold in the engine used.
Results are appended per question (resumable).
Usage: trackb_cont.py --port P --model-dir HF_DIR --data-dir D --ids IDS --ref gen_V.jsonl --out cont_V.jsonl
"""
import argparse
import json
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

from det_check import build_prompt_ids, load_questions

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--model-dir", required=True)
ap.add_argument("--data-dir", required=True)
ap.add_argument("--ids", required=True)
ap.add_argument("--ref", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-tokens", type=int, default=1024)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model_dir)
qs = load_questions(Path(args.data_dir), args.ids)
ref = {json.loads(l)["id"]: json.loads(l) for l in open(args.ref, encoding="utf-8")}
out = Path(args.out)
done = {json.loads(l)["id"] for l in open(out, encoding="utf-8")} if out.exists() else set()
url = f"http://127.0.0.1:{args.port}/completion"
with open(out, "a", encoding="utf-8") as f:
    for rec in qs:
        if rec["id"] in done or rec["id"] not in ref:
            continue
        p = build_prompt_ids(tok, rec)
        y = ref[rec["id"]]["out_ids"]
        rows = []
        for frac in (1 / 3, 2 / 3):
            t = max(1, int(len(y) * frac))
            body = {"prompt": p + y[:t], "n_predict": args.max_tokens - t, "temperature": 0.0, "seed": 0,
                    "cache_prompt": False, "return_tokens": True, "n_probs": 0}
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            res = json.loads(urllib.request.urlopen(req, timeout=600).read())
            c = res.get("tokens") or []
            rest = y[t:]
            fd = next((k for k in range(min(len(c), len(rest))) if c[k] != rest[k]), None)
            if fd is None and len(c) != len(rest):
                fd = min(len(c), len(rest))
            rows.append({"t": t, "equal": c == rest, "first_diff": fd, "len_ref": len(rest), "len_cont": len(c)})
        f.write(json.dumps({"id": rec["id"], "n": len(y), "checks": rows}) + "\n")
        f.flush()
print("TRACKB_CONT_DONE")
