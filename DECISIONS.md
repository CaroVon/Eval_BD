# Decisions log

Protocol invariants (plan v4.0, (i)-(iv)) are unchanged. Entries below are work-plan choices,
except D4, which changes the execution configuration and is recorded with its evidence.

## D1 (2026-10-28) Pilot parent: Qwen/Qwen2.5-3B-Instruct @ aa8e725
Non-thinking model, mature support in vLLM and llama.cpp, fast enough on A100 to iterate.

## D2 (2026-10-28) GPU: A100 80GB PCIe only
All A100 nodes are the same PCIe variant; H100/H200 nodes mix SXM, NVL and PCIe. On SM80 vLLM's
batch-invariant mode replaces cuBLAS matmuls with Triton persistent kernels, while on SM90 it uses
cuBLAS with split-k disabled, so the numerical path differs between A100 and H100. Truth matrix and
certificates must not mix GPU types. All jobs use `--gres=gpu:a100:1 --constraint=a100`.

## D3 (2026-10-29) No imatrix in the pilot ladder
Plan requires imatrix calibration text disjoint from the benchmarks; choosing it is a separate
decision. The pilot ladder (Q8_0, Q6_K, Q4_K_M, Q3_K_M) is quantized without imatrix; rerunning with
imatrix only repeats the quantize step.

## D4 (2026-10-30) Execution dtype float32 instead of 16-bit arithmetic
Plan sec 2.1 specified 16-bit arithmetic for track A. Pilot evidence (GSM8K, 200 questions):

- With bf16, exact logit ties between the parent token and another token are frequent
  (parent: 470 tie positions in 171/200 questions). vLLM greedy decoding breaks ties by smallest
  token id (verified: 0 rule violations across parent and 4 children), so a child flips a tie
  whenever the other tied token has a smaller id. These divergences come from bf16 logit
  resolution, not from quantization.
- With fp32, tie positions drop to 0 for all models. Q8_0 |A|/|P| falls from 0.717 to 0.478,
  median t\* rises from 65 to 94, continuation decode saving rises from 49% to 66%.
- fp32 also represents Q8_0 dequantized weights exactly.
- fp32 passes all four determinism layers (attention backend TRITON_ATTN, since FlashAttention has
  no fp32 path). Cost: decode about 1.5x slower than bf16 on A100.

Track A therefore certifies the model defined by the quantized weights under fp32 arithmetic.
The gap to deployed (quantized-kernel) execution is still measured by track B.

## D5 (2026-10-30) Tie handling in screening
A position counts as a divergence iff the predicted decoded token (smallest id among the tied
maximisers in the top-20 prompt logprobs) differs from the parent token. The prediction is checked
against the child's actual generation for every question (`tie_rule_violations`). Under fp32 no
ties occurred; the production screen can use a smaller top-k, to be re-measured for cost.
