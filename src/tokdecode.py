"""Token-id -> text decoding without transformers, for byte-level BPE (Qwen) and
SentencePiece-style vocabularies (Phi-3.5 / Llama: '▁' word prefix, <0xNN> byte fallback).

load(path) accepts a vocab.json (token -> id) or a tokenizer.json. Returns text(ids) -> str.
"""
import json
import re


def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


BYTE_DEC = {v: k for k, v in _bytes_to_unicode().items()}
_HEX = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


def load(path):
    d = json.load(open(path, encoding="utf-8"))
    is_tok_json = isinstance(d.get("model"), dict)  # a vocab.json may contain the token "model"
    vocab = d["model"]["vocab"] if is_tok_json else d
    inv = {v: k for k, v in vocab.items()}
    if is_tok_json:
        for t in d.get("added_tokens", []):
            if t.get("special", False):
                inv[t["id"]] = ""  # special tokens decode to nothing (some vocabs also list them)
            else:
                inv.setdefault(t["id"], t["content"])
    sp = sum(1 for k in vocab if k.startswith("▁")) > sum(1 for k in vocab if k.startswith("Ġ"))

    if not sp:
        def text(ids):
            s = "".join(inv.get(int(t), "") for t in ids)
            return bytearray(BYTE_DEC[c] for c in s if c in BYTE_DEC).decode("utf-8", errors="replace")
        return text

    def text(ids):
        out = bytearray()
        for t in ids:
            p = inv.get(int(t), "")
            m = _HEX.match(p)
            if m:
                out.append(int(m.group(1), 16))
            elif p.startswith("<") and p.endswith(">") and len(p) > 2 and p not in ("<unk>",):
                continue  # control tokens such as <s>, </s>
            else:
                out += p.replace("▁", " ").encode("utf-8")
        return out.decode("utf-8", errors="replace")
    return text
