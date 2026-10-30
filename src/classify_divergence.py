"""Analysis 2: what happens at and after the first divergence t*.

For each question in D (child greedy output differs from the parent output):
  token class at t*  opener (t* = 0), whitespace/newline, digit, punctuation/markup, word,
                     plus 'split' when one token is a prefix of the other (tokenization only)
  outcome            resync: a common block of >= W tokens starts within R tokens of t* in both
                     outputs (local edit, then the texts realign)
                     same answer / changed answer (extracted with the task grader)
Usage: classify_divergence.py <truth_dir> <vocab.json> <test.jsonl>
"""
import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tasks import answer  # noqa: E402
import tokdecode  # noqa: E402

W, RWIN = 8, 30


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


BYTE_DEC = {v: k for k, v in bytes_to_unicode().items()}


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def main():
    truth, vocab_path, test_path = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    text = tokdecode.load(vocab_path)
    recs = {r["id"]: r for r in read_jsonl(test_path)}
    gen = {r["id"]: r for r in read_jsonl(truth / "parent_gen" / "gen_A.jsonl")}

    def tclass(t, other, pos):
        if pos == 0:
            return "opener"
        a, b = text([t]), text([other])
        if a and b and (a.startswith(b) or b.startswith(a)):
            return "split"
        if a.strip() == "":
            return "space/newline"
        if re.search(r"\d", a):
            return "digit"
        if not re.search(r"[A-Za-z]", a):
            return "punct/markup"
        return "word"

    out = {}
    print(f"== {truth}")
    for ver in ["Q8_0", "Q6_K", "Q4_K_M", "Q3_K_M"]:
        rows = []
        for r in read_jsonl(truth / "screen" / ver / "records.jsonl"):
            if not r["in_D"]:
                continue
            t = r["t_star"]
            y0, yq = gen[r["id"]]["out_ids"], r["child_out_ids"]
            ps, cs = y0[t:], yq[t:]
            cls = tclass(ps[0], cs[0], t) if ps and cs else "end"
            sm = SequenceMatcher(None, ps[:1500], cs[:1500], autojunk=False)
            blocks = [b for b in sm.get_matching_blocks() if b.size >= W]
            resync = any(b.a <= RWIN and b.b <= RWIN for b in blocks)
            same_ans = answer(text(yq), recs[r["id"]]) == answer(gen[r["id"]]["text"], recs[r["id"]])
            rows.append({"cls": cls, "resync": resync, "same_ans": same_ans, "ratio": sm.ratio(),
                         "harm": r["parent_correct"] and not r["child_correct"], "inP": r["parent_correct"],
                         "pair": (text([ps[0]]) if ps else "", text([cs[0]]) if cs else "")})
        n = len(rows)
        cls_c = Counter(x["cls"] for x in rows)
        print(f"{ver:7} |D|={n}  same answer {sum(x['same_ans'] for x in rows) / n:.3f}  "
              f"resync within {RWIN} tok {sum(x['resync'] for x in rows) / n:.3f}  "
              f"median text similarity {sorted(x['ratio'] for x in rows)[n // 2]:.2f}")
        for c, k in cls_c.most_common():
            sub = [x for x in rows if x["cls"] == c]
            subP = [x for x in sub if x["inP"]]
            print(f"     {c:14} {k / n:5.1%}  resync {sum(x['resync'] for x in sub) / k:.2f}  "
                  f"same-ans {sum(x['same_ans'] for x in sub) / k:.2f}  "
                  f"harm|P {sum(x['harm'] for x in subP)}/{len(subP)}")
        # outcome classes among A (parent-correct divergent questions)
        A = [x for x in rows if x["inP"]]
        rs = [x for x in A if x["resync"]]
        nrs = [x for x in A if not x["resync"]]
        print(f"     A={len(A)}: resync {len(rs)} (harm {sum(x['harm'] for x in rs)}), "
              f"no-resync same-answer {sum(x['same_ans'] for x in nrs)} (harm {sum(x['harm'] and x['same_ans'] for x in nrs)}), "
              f"no-resync changed-answer {sum(not x['same_ans'] for x in nrs)} (harm {sum(x['harm'] and not x['same_ans'] for x in nrs)})")
        print("     top t* token pairs (parent | child):",
              [(f"{a!r}|{b!r}", k) for (a, b), k in Counter(x["pair"] for x in rows).most_common(6)])
        out[ver] = {"n_D": n, "classes": dict(cls_c),
                    "same_answer": sum(x["same_ans"] for x in rows) / n,
                    "resync": sum(x["resync"] for x in rows) / n,
                    "A": len(A), "A_resync": len(rs), "A_resync_harm": sum(x["harm"] for x in rs),
                    "A_nores_same": sum(x["same_ans"] for x in nrs),
                    "A_nores_changed": sum(not x["same_ans"] for x in nrs),
                    "A_harm": sum(x["harm"] for x in A)}
    (truth / "divergence_classes.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
