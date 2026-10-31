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
Full GSM8K runs use top-2 (enough to detect every tie: two equal maxima are always visible);
top-1/top-2 logprobs add ~17% over a bare prefill, top-20 adds ~140%.

## D6 (2026-10-30) MMLU-Pro subsets
TIGER-Lab/MMLU-Pro test @ b189ec7 (12,032 questions, 14 categories). A 3,000-question subset with
equal allocation across categories (about 214 each, so category slices have comparable size) and
a 1,000-question pilot nested in it (first 71-72 of each category's shuffled list), seed 20261030.
Zero-shot CoT prompt in the official MMLU-Pro format ending with "The answer is (X)"; grading
uses the official extraction order (`src/tasks.py`).

## D7 (2026-10-30) Screening scope and lazy prefill
The certificate bounds the harm rate, which only involves parent-correct questions, so only P
needs screening. Screening prefill can stop at the first divergence (t\* + 1 parent tokens),
since D and t\* are fixed under determinism. Both reduce screening cost without changing any
certificate; `src/simulate_cert.py --scope P --lazy 1` accounts for them. The current screening
job still prefills the full parent output for all questions (needed to verify lemma 3); a
production screen would implement the chunked early stop.

## D8 (2026-10-30) Exploratory second parent: microsoft/Phi-3.5-mini-instruct @ 2fe1924
Chosen only to test whether the margin scaling law (RESULTS.md sec. 13) holds for a different
family: MIT license, not gated, 3.8B, phi3 architecture supported by llama.cpp, transformers GGUF
loading and vLLM. This is not the plan's final choice of second/third parent, which is still to be
made after G1. Same ladder, no imatrix, fp32 execution, A100 only.

## D9 (2026-10-30) Net-degradation certificate kept as an extension
`docs/net_certificate.md`. It certifies theta = H - Rep instead of the harm rate H; it is valid
for any configuration and reduces to the plan's certificate when B is not audited. Reported
alongside the harm certificate, not as a replacement, until its design and lower bound are worked out.

## D10 (2026-10-30) Execution mode is per model family; Phi-3.5 runs eager
Phi-3.5-mini fails the four-layer check in compiled mode (5/200 questions differ between
processes) and passes with `--enforce-eager`. Qwen2.5-3B passes in compiled mode (three separate
checks). Rule: every parent gets its own four-layer check in the exact mode used for its truth
matrix, and the mode (compiled / eager) is part of the recorded execution configuration.

## D11 (2026-10-30) More exploratory parents for the margin law
HuggingFaceTB/SmolLM2-1.7B-Instruct @ 31b70e2 (Apache-2.0, llama architecture) as a third family,
and Qwen/Qwen2.5-7B-Instruct @ a09a354 (Apache-2.0) to test model scale within the Qwen family and
the prefill/decode cost ratio at 7B. Both run in eager mode after their own four-layer check
(D10). Like D8, these are not the plan's final parent choices.

## D12 (2026-10-30) Qwen2.5-14B and Llama-3.1-8B as further exploratory parents
Qwen/Qwen2.5-14B-Instruct @ cf98f3b (Apache-2.0) tests scale beyond 7B and a KV-limited decode
batch (fp32 weights about 59 GB on an 80 GB A100). meta-llama/Llama-3.1-8B-Instruct @ 0e9e39f
(Llama 3.1 Community License, gated; access granted to the user's HF account) is a fourth family.
Both use the same llama.cpp ladder without imatrix, fp32 dequantization and eager mode after their own
four-layer check. The HF token is kept in ~/.config/huggingface/token (HF_TOKEN_PATH), outside the
group-readable project space. Exploratory, like D8 and D11.
