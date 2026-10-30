"""Dequantize a GGUF quant back to a dense HF checkpoint (bf16 or fp32) (track A, plan sec 3.1).

The architecture config and tokenizer are copied from the reference HF checkpoint so
the only difference from the parent is the weights. Writes MANIFEST.json with per-tensor
relative error against the parent weights as a sanity check.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
import transformers
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

COPY_FROM_REF = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                 "vocab.json", "merges.txt", "LICENSE", "PINNED_REVISION.json",
                 "tokenizer.model", "special_tokens_map.json", "added_tokens.json"]


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def load_ref_tensors(ref_dir):
    index = ref_dir / "model.safetensors.index.json"
    if index.exists():
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:  # single-file checkpoint
        shards = ["model.safetensors"]
    out = {}
    for shard in shards:
        with safe_open(ref_dir / shard, framework="pt") as f:
            for k in f.keys():
                out[k] = f.get_tensor(k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--ref", required=True, help="parent HF checkpoint dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--quant", required=True)
    ap.add_argument("--llamacpp-pin", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = ap.parse_args()
    gguf, ref, out = Path(args.gguf), Path(args.ref), Path(args.out)
    dtype = getattr(torch, args.dtype)

    model = AutoModelForCausalLM.from_pretrained(str(gguf.parent), gguf_file=gguf.name,
                                                 dtype=dtype)
    sd = {k: v.detach() for k, v in model.state_dict().items()}
    ref_sd = load_ref_tensors(ref)

    tied = json.loads((ref / "config.json").read_text()).get("tie_word_embeddings", False)
    if tied:
        sd.pop("lm_head.weight", None)
    missing = sorted(set(ref_sd) - set(sd))
    extra = sorted(set(sd) - set(ref_sd))
    shape_bad = [k for k in ref_sd if k in sd and tuple(sd[k].shape) != tuple(ref_sd[k].shape)]
    if missing or extra or shape_bad:
        raise SystemExit(f"tensor mismatch: missing={missing[:5]} extra={extra[:5]} shape={shape_bad[:5]}")

    rel = {}
    for k, w in ref_sd.items():
        a, b = sd[k].float(), w.float()
        rel[k] = ((a - b).norm() / b.norm().clamp_min(1e-12)).item()

    out.mkdir(parents=True, exist_ok=True)
    # save_pretrained refuses GGUF-flagged models, so write the dense weights directly.
    save_file({k: sd[k].to(dtype).contiguous() for k in ref_sd},
              out / "model.safetensors", metadata={"format": "pt"})
    for name in COPY_FROM_REF:  # parent config/tokenizer, so only the weights differ
        if (ref / name).exists():
            shutil.copy2(ref / name, out / name)
    cfg = json.loads((out / "config.json").read_text())
    cfg["torch_dtype"] = args.dtype
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    vals = sorted(rel.values())
    manifest = {
        "quant": args.quant, "gguf": str(gguf), "gguf_sha256": sha256(gguf),
        "llamacpp": Path(args.llamacpp_pin).read_text().strip(),
        "parent": json.loads((ref / "PINNED_REVISION.json").read_text()),
        "dequant_dtype": args.dtype, "imatrix": None,
        "transformers": transformers.__version__, "torch": torch.__version__,
        "rel_err_median": vals[len(vals) // 2], "rel_err_max": vals[-1],
        "rel_err_per_tensor": rel,
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(f"DEQUANT_DONE {args.quant} median_rel_err={manifest['rel_err_median']:.4g} "
          f"max_rel_err={manifest['rel_err_max']:.4g}")


if __name__ == "__main__":
    main()
