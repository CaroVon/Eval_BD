"""Track B pilot: greedy generation with the deployed GGUF under llama.cpp (llama-server, one slot).

Sends exactly the Track A prompt token ids (same chat template, build_prompt_ids) to /completion
with temperature 0 and n_predict = Lmax, and records the generated token ids and text. Results are
appended one question at a time, so a restarted job skips finished questions.
Usage: trackb_gen.py --port P --model-dir HF_DIR --data-dir D --ids IDS --out out.jsonl [--max-tokens 1024]
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

from det_check import build_prompt_ids, load_questions
from tasks import answer, grade

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--model-dir", required=True, help="HF dir for the tokenizer / chat template")
ap.add_argument("--data-dir", required=True)
ap.add_argument("--ids", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-tokens", type=int, default=1024)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model_dir)
qs = load_questions(Path(args.data_dir), args.ids)
out = Path(args.out)
done = set()
if out.exists():
    done = {json.loads(l)["id"] for l in open(out, encoding="utf-8")}
url = f"http://127.0.0.1:{args.port}/completion"
t0 = time.time()
with open(out, "a", encoding="utf-8") as f:
    for rec in qs:
        if rec["id"] in done:
            continue
        ids = build_prompt_ids(tok, rec)
        body = {"prompt": ids, "n_predict": args.max_tokens, "temperature": 0.0, "seed": 0,
                "cache_prompt": False, "return_tokens": True, "n_probs": 0}
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        res = json.loads(urllib.request.urlopen(req, timeout=600).read())
        text = res.get("content", "")
        f.write(json.dumps({"id": rec["id"], "prompt_len": len(ids), "out_ids": res.get("tokens"),
                            "text": text, "answer": answer(text, rec), "correct": grade(text, rec),
                            "stop": res.get("stop_type"), "tokens_predicted": res.get("tokens_predicted")},
                           ensure_ascii=False) + "\n")
        f.flush()
print("TRACKB_DONE", len(qs), "seconds", round(time.time() - t0, 1))
