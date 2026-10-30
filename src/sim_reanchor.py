"""Cost of exact child evaluation by verifying the parent output as a draft (prompt-lookup style).

Because the child's greedy output y_q is known from the truth matrix, the verifier's behaviour
is replayed exactly:
  - anchor: the parent position after the most recent occurrence (within the parent output) of
    the child's last n generated tokens; at the start the anchor is position 0
  - round with an anchor: verify a draft of L parent tokens in one forward pass (cost r per
    draft token, KV of accepted tokens kept), accept the matching prefix, and get the child's
    next token for free from the same pass (bonus token, as in speculative decoding)
  - round without an anchor: decode one token (cost 1)
Output equals y_q by construction (lemma 3 / prefill-decode invariance), so the result is exact.
Each round is one sequential step; the number of rounds bounds latency, not cost.
"""
import json
import sys
from pathlib import Path


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def replay(y0, yq, L, n, r):
    index = {}
    for j in range(len(y0) - n + 1):
        index.setdefault(tuple(y0[j:j + n]), []).append(j + n)
    pos, anchor, cost, rounds, drafted, accepted = 0, 0, 0.0, 0, 0, 0
    while pos < len(yq):
        rounds += 1
        draft = y0[anchor:anchor + L] if anchor is not None else []
        if draft:
            a = 0
            while a < len(draft) and pos + a < len(yq) and draft[a] == yq[pos + a]:
                a += 1
            cost += r * (len(draft) + 1)  # draft positions + the bonus position
            drafted += len(draft)
            accepted += a
            nxt = anchor + a  # parent position the bonus token would match
            pos += a + 1  # accepted prefix + bonus token from the same pass
            # keep the anchor only if the whole draft and the bonus token agree with the parent
            ok = a == len(draft) and nxt < len(y0) and pos - 1 < len(yq) and yq[pos - 1] == y0[nxt]
            anchor = nxt + 1 if ok else None
        else:
            cost += 1.0
            pos += 1
            anchor = None
        if anchor is None and pos >= n:
            cands = index.get(tuple(yq[pos - n:pos]))
            if cands:
                anchor = cands[-1] if len(cands) == 1 else min(cands, key=lambda c: abs(c - pos))
        if anchor is not None and anchor >= len(y0):
            anchor = None
    return cost, rounds, drafted, accepted


def main():
    truth, r = Path(sys.argv[1]), float(sys.argv[2])
    par = {g["id"]: g["out_ids"] for g in read_jsonl(truth / "parent_gen" / "gen_A.jsonl")}
    res = {}
    print(f"== {truth}  r={r}  (cost unit: one decode token; speedup = child tokens / cost)")
    for v in ["Q8_0", "Q6_K", "Q4_K_M", "Q3_K_M"]:
        recs = read_jsonl(truth / "screen" / v / "records.jsonl")
        res[v] = {}
        for L in (16, 32, 64):
            for n in (2, 4):
                tot = cost = rounds = drafted = acc = 0
                for rec in recs:
                    c, k, dr, ac = replay(par[rec["id"]], rec["child_out_ids"], L, n, r)
                    tot += len(rec["child_out_ids"]); cost += c; rounds += k; drafted += dr; acc += ac
                res[v][f"L{L}_n{n}"] = {"speedup": tot / cost, "rounds_per_token": rounds / tot,
                                        "draft_acceptance": acc / max(drafted, 1)}
        best = max(res[v].items(), key=lambda kv: kv[1]["speedup"])
        print(f"  {v:7} best {best[0]}: speedup {best[1]['speedup']:.2f}x, "
              f"acceptance {best[1]['draft_acceptance']:.2f}, sequential rounds per token "
              f"{best[1]['rounds_per_token']:.3f}  | all: "
              + " ".join(f"{k}:{x['speedup']:.2f}" for k, x in res[v].items()))
    (truth / f"reanchor_r{r}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
