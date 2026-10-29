# Eval_BD: Structure-aware finite-population certification of quantized LLM release ladders

Code for research plan v4.0 (target: ICML 2027). Track A pilot on the UChicago DSI cluster.

## Layout

| Path | Contents |
|---|---|
| `src/det_check.py` | Four-layer determinism check (plan sec 2.2): repeat run, prefill vs decode, continuation, batch sensitivity |
| `src/det_report.py` | Summarises a determinism run into `report.json` |
| `src/dequant_gguf.py` | GGUF quant -> dense bf16/fp32 safetensors, parent config/tokenizer, per-tensor error manifest |
| `src/child_screen.py` | Exact screening of one child along parent outputs: D, t\*, lemma 3 checks, tie-rule check, costs |
| `src/analyze_pilot.py` | Breakdown of screening results |
| `scripts/*.sbatch` | Slurm jobs (A100 only, `--constraint=a100`) |
| `scripts/stage_env.sh` | Unpacks the conda-packed env onto node-local NVMe (`--gres=local:NG`) |
| `scripts/prep_gsm8k.py`, `scripts/dl_model.py` | Data and model download at pinned revisions |
| `results/` | Small result files copied from the cluster outputs (summaries only) |
| `DECISIONS.md` | Protocol and work-plan decisions with reasons |
| `RESULTS.md` | Pilot results so far |

Large artefacts stay on the cluster under `/net/spaces/jevans/klab/caroyf/` (`data/ICML`, `outputs/ICML`, `logs/ICML`).
`third_party/llama.cpp` is not committed; the pinned commit is in `results/pins/llama.cpp.PINNED`.

## Environment

- vLLM 0.32.0, torch 2.13.0+cu130, transformers (see `results/pins/qc-vllm.txt`)
- llama.cpp `eb4e244`, built with CUDA 13.1 for sm_80
- GPU: NVIDIA A100 80GB PCIe, driver 595.84
- `VLLM_BATCH_INVARIANT=1`, `async_scheduling=False`, `enable_prefix_caching=False`
- Execution dtype: **float32** (see DECISIONS.md D4)

The conda env lives on Lustre but must not be imported from there on compute nodes (tens of
thousands of small files stall in `cl_sync_io_wait`). Jobs unpack `envs_pack/qc-vllm.tar` to
`/local/scratch` via `scripts/stage_env.sh`.
