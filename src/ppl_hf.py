"""WikiText-2 perplexity of an HF checkpoint, computed the way llama.cpp's llama-perplexity does.

llama-perplexity tokenizes the whole file once (adding BOS if the vocab asks for it), cuts it into
non-overlapping chunks of n_ctx tokens, puts BOS at the first position of each chunk when add_bos,
and scores the predictions for positions n_ctx/2 .. n_ctx-2 (the next token at n_ctx/2+1 .. n_ctx-1).
Matching that lets the dequantized safetensors (Track A) be compared with the GGUF they came from
run by llama.cpp's own kernels.
Usage: ppl_hf.py --model DIR --text test.raw --out result.json [--ctx 512] [--dtype float32]
"""
import argparse
import json
import math
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--text", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--ctx", type=int, default=512)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--dtype", default="float32")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
text = open(args.text, encoding="utf-8").read()
ids = tok(text, add_special_tokens=False)["input_ids"]
probe = tok("a")["input_ids"]
add_bos = tok.bos_token_id is not None and len(probe) > 0 and probe[0] == tok.bos_token_id
if add_bos:
    ids = [tok.bos_token_id] + ids
n_ctx = args.ctx
n_chunk = len(ids) // n_ctx
first = n_ctx // 2

model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).cuda().eval()
nll, count, t0 = 0.0, 0, time.time()
with torch.no_grad():
    for b in range(0, n_chunk, args.batch):
        rows = []
        for c in range(b, min(b + args.batch, n_chunk)):
            row = ids[c * n_ctx:(c + 1) * n_ctx]
            if add_bos:
                row = [tok.bos_token_id] + row[1:]
            rows.append(row)
        x = torch.tensor(rows, device="cuda")
        logits = model(x).logits.float()
        lp = torch.log_softmax(logits[:, first:n_ctx - 1], dim=-1)
        tgt = x[:, first + 1:n_ctx]
        nll -= lp.gather(-1, tgt.unsqueeze(-1)).sum().item()
        count += tgt.numel()
ppl = math.exp(nll / count)
res = {"model": args.model, "tokens": len(ids), "chunks": n_chunk, "n_ctx": n_ctx, "add_bos": add_bos,
       "scored": count, "nll": nll / count, "ppl": ppl, "dtype": args.dtype, "seconds": time.time() - t0}
json.dump(res, open(args.out, "w"), indent=1)
print("PPL_HF", json.dumps(res))
