"""Replay vLLM continuous batching for one generation run to estimate the effective decode batch.

Approximates the V1 scheduler: requests are admitted in submission order while their prompt fits in the
free KV blocks and fewer than max_num_seqs are running; every step each running request appends one token
(allocating a new block every BLOCK tokens); on overflow the newest running request is preempted and later
recomputed from scratch. Reports decode steps, mean batch per step and preemptions, and the per-step time
implied by the measured decode wall time.
Usage: concurrency_sim.py <gen_A.jsonl> <kv_tokens> <max_num_seqs> [decode_seconds]
"""
import json
import sys
from collections import deque

BLOCK = 16
gen = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
kv_blocks = int(sys.argv[2]) // BLOCK
max_seqs = int(sys.argv[3])
dec_s = float(sys.argv[4]) if len(sys.argv) > 4 else None

blocks = lambda n: -(-n // BLOCK)
waiting = deque((g["prompt_len"], len(g["out_ids"])) for g in gen)
running = []  # [prompt_len, out_len, generated]
used = steps = batch_sum = preempt = 0
hist = []
while waiting or running:
    while waiting and len(running) < max_seqs and used + blocks(waiting[0][0] + 1) <= kv_blocks:
        p, o = waiting.popleft()
        running.append([p, o, 0])
        used += blocks(p + 1)
    if not running:
        raise SystemExit("a single request does not fit in the KV cache")
    steps += 1
    batch_sum += len(running)
    hist.append(len(running))
    for r in running:
        r[2] += 1
    # grow allocations; preempt newest requests on overflow
    need = sum(blocks(r[0] + r[2] + 1) for r in running)
    while need > kv_blocks:
        p, o, _ = running.pop()
        waiting.appendleft((p, o))
        preempt += 1
        need = sum(blocks(r[0] + r[2] + 1) for r in running)
    done = [r for r in running if r[2] >= r[1]]
    running = [r for r in running if r[2] < r[1]]
    used = sum(blocks(r[0] + r[2] + 1) for r in running)

tok = sum(len(g["out_ids"]) for g in gen)
res = {"requests": len(gen), "decode_tokens": tok, "steps": steps, "mean_batch": batch_sum / steps,
       "token_weighted_batch": sum(b * b for b in hist) / batch_sum, "preemptions": preempt,
       "kv_tokens": kv_blocks * BLOCK, "max_num_seqs": max_seqs}
if dec_s:
    res["ms_per_step"] = 1000 * dec_s / steps
print(json.dumps(res))
