# Pilot results (Qwen2.5-3B-Instruct, GSM8K test, 200-question subset)

Subset: 200 ids drawn with seed 20261028 (`results/data/determinism_subset_ids.json`).
Greedy decoding, Lmax = 1024, A100 80GB PCIe, vLLM 0.32.0 batch-invariant mode.

## 1. Determinism (plan sec 2.2, RQ2)

| Layer | bf16, BI on | bf16, BI off (control) | fp32, BI on |
|---|---|---|---|
| 1 repeat run, separate processes | 0/200 differ | 26/200 differ (2 answers) | 0/200 |
| 2 prefill top-1 vs decoded token | 0 / 57,475 tokens | 81 tokens in 63 questions (0.14%) | 0 / 57,387 |
| 3 continuation from 1/3, 2/3 cut | 0/400 differ | 131/400 differ | 0/400 |
| 4 max_num_seqs 256 -> 8, shuffled | 0/200 differ | 131/200 differ (9 answers) | 0/200 |

The control shows the checks are sensitive: without batch invariance, 4.5% of answers flip just
from changing the batch.

## 2. Exact screening, fp32 (adopted config)

| Version | Child acc. | \|A\|/\|P\| | median t\* | continuation decode saving | lemma 3 violations | flips outside D |
|---|---|---|---|---|---|---|
| parent (control) | 78.5% | 0 | - | - | 0 | 0 |
| Q8_0 | 79.0% | **0.478** | 94 | 66% | 0 | 0 |
| Q6_K | 80.0% | 0.873 | 29 | 26% | 0 | 0 |
| Q4_K_M | 74.0% | 1.000 | 0 | 5% | 0 | 0 |
| Q3_K_M | 41.5% | 1.000 | 0 | 0.5% | 0 | 0 |

|P| = 157. Rough 95% interval for Q8_0 |A|/|P|: 0.40 to 0.56.

## 3. Exact screening, bf16 (superseded, see DECISIONS.md D4)

| Version | \|A\|/\|P\| | median t\* | saving | tie positions |
|---|---|---|---|---|
| parent | 0 | - | - | 470 |
| Q8_0 | 0.717 | 65 | 49% | 453 |
| Q6_K | 0.918 | 30 | 25% | 431 |
| Q4_K_M | 0.994 | 0 | 5% | 401 |
| Q3_K_M | 1.000 | 0 | 0.5% | 509 |

## 4. Dequantization sanity (relative Frobenius error vs parent weights, median / max)

| Quant | BPW | bf16 dequant |
|---|---|---|
| Q8_0 | 8.50 | 0.57% / 0.96% |
| Q6_K | 6.56 | 1.8% / 2.9% |
| Q4_K_M | 4.99 | 2.9% / 8.7% |
| Q3_K_M | 4.11 | 7.2% / 16% |

## 5. Cost observations

- bf16 decode of 200 questions: 17 s (batch 256); prefill along parent outputs with top-1 logprobs: 3.5 s.
- fp32 decode: 26 s. fp32 prefill with top-20 logprobs (tie check): 12 s; needs re-measuring with small top-k.

## 6. Full GSM8K test set (1,319), fp32

| Version | Child acc. | \|A\|/\|P\| | median t\* | harm / repair flips | lemma 3 violations | ties |
|---|---|---|---|---|---|---|
| parent | 82.4% | 0 | - | - | 0 | 0 |
| Q8_0 | 83.0% | **0.519** | 94 | 19 / 27 | 0 | 0 |
| Q6_K | 82.0% | 0.902 | 31 | 57 / 51 | 0 | 0 |
| Q4_K_M | 79.5% | 0.999 | 0 | 104 / 65 | 0 | 0 |
| Q3_K_M | 47.8% | 1.000 | 0 | 503 / 46 | 0 | 0 |

|P| = 1,087; 95% interval for Q8_0 |A|/|P| about 0.49 to 0.55. No flips outside D.

## 7. Prefill vs decode cost, fp32, A100 (E3)

| Measurement | per-token prefill / decode |
|---|---|
| 200 questions, `prefill_cost.py` (top-2) | 0.20 |
| 1,319 questions, screening job (top-2) | **0.385** |

Decode is cheaper per token at the larger batch, so 0.385 is the representative value. One
full prefill along the parent outputs costs about 53% of a full generation (45 s vs 85 s).
Bare prefill / top-1 / top-2 / top-20 on 200 questions: 5.8 / 6.8 / 6.9 / 14.1 s.

## 8. Certificate simulation on the full GSM8K truth matrix

`src/simulate_cert.py`, 2,000 Monte Carlo reps, independent Poisson sampling per version,
slice-cover design (prop. 10), exact binomial certificate (thm. 4) at delta / |S|.
Slices: all + quantile bins of parent output length and prompt length.

tau = 0.05, delta = 0.05, 9 slices (>= 100 questions), |S| = 36: simultaneous coverage 0.98.

| Method (r = 0.385) | speedup vs full evaluation of all 4 versions |
|---|---|
| ours, screening all questions, full prefill | 1.50 |
| ours, screening P only, lazy prefill (D7) | 2.66 |
| ours on Q8_0 only, B1 elsewhere | 3.44 |
| **B1: no screening, audit full generations on g n P** | **3.89** |

Screening must touch every parent-correct question, while B1 touches only the sampled ones
(pi about 0.33 here), so B1 is cheaper at this granularity. Cost of ours relative to B1 on Q8_0
as the audit rate rises (scope P, lazy):

| Setting | mean pi | Q8_0 cost ours / B1, r = 0.385 | r = 0.2 |
|---|---|---|---|
| tau 0.05, 9 slices | 0.33 | 1.52 | 0.99 |
| tau 0.05, 17 slices | 0.58 | 1.05 | 0.72 |
| tau 0.02, 17 slices | 0.84 | 0.85 | 0.61 |
| tau 0.02, 33 slices | 0.98 | 0.79 | 0.58 |

Structure pays off only at high audit rates (fine-grained simultaneous certification), and the
break-even point is set by the prefill/decode cost ratio r. Worst-slice certificates are the same
for both methods at equal pi; the difference is cost only.

## 9. Prefill vs decode cost, same 1,319 questions on two GPUs (fp32 Q8_0, top-2)

| GPU | decode | prefill (top-2) | per-token r | bare prefill r |
|---|---|---|---|---|
| A100 80GB PCIe | 90.5 s / 369,637 tok | 45.6 s / 509,901 tok | **0.365** | 0.308 |
| H100 NVL (cost only, not truth data) | 51.6 s | 20.6 s | **0.290** | 0.240 |

Top-20 logprobs cost 68.5 s (A100) and 72.7 s (H100); top-1/top-2 cost about 17% over bare.

## 10. MMLU-Pro pilot (1,000 questions, 14 categories x 71-72), fp32, A100

Parent output length: mean 466, median 407 tokens (GSM8K: 287); 72 outputs (7.2%) hit Lmax = 1024.

| Version | Child acc. | \|A\|/\|P\| | median t\* | harm / repair flips | lemma 3 violations | ties |
|---|---|---|---|---|---|---|
| parent | 44.6% | 0 | - | - | 0 | 0 |
| Q8_0 | 43.9% | **0.756** | 88 | 53 / 46 | 0 | 0 |
| Q6_K | 44.5% | 0.980 | 34 | 84 / 83 | 0 | 0 |
| Q4_K_M | 43.0% | 1.000 | 15 | 102 / 86 | 0 | 0 |
| Q3_K_M | 23.9% | 1.000 | 1 | 279 / 72 | 0 | 0 |

|P| = 446.

## 11. Certificate simulation with measured r (scope P, lazy, 2,000 reps)

Files: `results/truth/*/sim_<setting>_r<r>.json`. "ours/B1" is the expected Q8_0 audit cost ratio
(< 1 means exact screening is cheaper). Speedups are over full evaluation of all four versions.

| Dataset, setting | slices | mean pi | speedup ours / sel / B1 (A100) | Q8_0 ours/B1 A100 | H100 | coverage | worst-slice UCB Q8_0 |
|---|---|---|---|---|---|---|---|
| GSM8K, tau .05, quartiles | 9 | 0.33 | 2.72 / 3.48 / 3.89 | 1.47 | 1.26 | 0.984 | 0.102 |
| GSM8K, tau .05, octiles | 17 | 0.58 | 1.91 / 2.19 / 2.20 | 1.01 | 0.88 | 0.989 | 0.121 |
| GSM8K, tau .02, octiles | 17 | 0.84 | 1.45 / 1.58 / 1.51 | 0.83 | 0.73 | 0.973 | 0.078 |
| MMLU-Pro, tau .05, quartiles | 9 | 0.40 | 4.24 / 5.58 / 6.21 | 1.43 | 1.27 | 0.982 | 0.182 |
| MMLU-Pro, tau .05, quartiles + categories (m = 50) | 23 | 0.85 | 2.53 / 2.90 / 2.91 | 1.01 | 0.93 | 0.975 | 0.203 |
| MMLU-Pro, tau .10, quartiles + categories (m = 50) | 23 | 0.61 | 3.21 / 3.89 / 4.05 | 1.16 | 1.05 | 0.988 | 0.278 |

No version is certified below tau in any setting: Q8_0 has real harm flips, so the worst-slice
upper bound stays above tau (0.10 on GSM8K, about 0.2 on MMLU-Pro at tau = 0.05).

## 12. Why |A|/|P| is high: hazard model and cascade check (`src/diagnose.py`)

A constant per-token divergence rate rho fits the screening data (MLE over teacher-forced
first-divergence times):

| Version | rho GSM8K | rho MMLU-Pro | GSM8K share in D by output-length quartile, obs / pred |
|---|---|---|---|
| Q8_0 | 0.0030 | 0.0040 | 0.42 0.48 0.63 0.65 / 0.41 0.51 0.58 0.69 |
| Q6_K | 0.0119 | 0.0167 | 0.84 0.91 0.94 0.96 / 0.88 0.94 0.97 0.99 |
| Q4_K_M | 0.0721 (0.038 after t = 0) | 0.0474 | all 1.00 |

So |A|/|P| ~ 1 - E[(1 - rho)^T]. On MMLU-Pro short outputs diverge more than predicted
(0.76 vs 0.61) and long ones less (0.80 vs 0.95): the rate varies across questions.

Flips inside A are rare: 19/564 (3.4%) for GSM8K Q8_0, 5.8% Q6_K, 15.7% MMLU-Pro Q8_0.
Harm vs repair flips are close to symmetric for high bit widths (GSM8K Q8_0 19/27, Q6_K 57/51;
MMLU-Pro Q8_0 53/46, Q6_K 84/83). MMLU-Pro Q8_0 has harm rate 5.3% > tau = 0.05 while the net
degradation is 0.7%, so a harm-only certificate cannot pass there at any audit rate.

Cascade drafts do not help: identical-output rate on P, Q6_K vs parent 0.098, Q6_K vs Q8_0 0.076
(GSM8K). Each llama.cpp quant is produced independently from bf16, so the perturbations are not
nested.

## 13. Analysis 1: margin profiles (`src/margin_profile.py`, `src/analyze_margins.py`)

Top-2 prompt logprobs along the parent outputs for parent and children (fp32, A100). m_t is the
parent margin of y0_t (top-1 minus top-2 logprob), g_t the child margin of y0_t, d_t = g_t - m_t.

Divergence happens only at parent near-ties (GSM8K Q8_0):

| parent margin | < 0.1 | 0.1-0.25 | 0.25-0.5 | 0.5-1 | 1-2 | > 2 |
|---|---|---|---|---|---|---|
| P(divergence at t) | 21% | 1.1% | 0.2% | 0.13% | 0.02% | ~0 |

1.2% (GSM8K) / 1.9% (MMLU-Pro) of positions have parent margin < 0.1.
The perturbation d_t is close to additive noise: its sd is flat across margin bins and
corr(d, m) is -0.07 to -0.11 for Q8_0 to Q4_K_M.

| Version | noise sd sigma (logits), GSM8K / MMLU-Pro |
|---|---|
| Q8_0 | 0.072 / 0.059 |
| Q6_K | 0.23 / 0.20 |
| Q4_K_M | 0.83 / 0.64 |
| Q3_K_M | not noise-like (corr(d, m) = -0.54) |

Q6_K / Q8_0 sigma ratio 3.2 equals their weight-error ratio (1.8% / 0.57%).

Scaling law: with Gaussian noise, rho ~ f_m(0) sigma / sqrt(2 pi), where f_m(0) is the density of
parent margins at 0.

| Dataset | f_m(0) | predicted rho/sigma | observed rho/sigma (Q8 / Q6 / Q4) |
|---|---|---|---|
| GSM8K | 0.12 | 0.048 | 0.040 / 0.044 / 0.037 |
| MMLU-Pro | 0.19 | 0.077 | 0.071 / 0.074 / 0.068 |

Screening power is therefore set by parent near-tie density x quantization noise x output length.
The question-level Gaussian model over-predicts D for Q8_0 (0.67 vs 0.55 on GSM8K), so the real
noise tail is lighter than Gaussian. By token class, 61% (GSM8K) / 79% (MMLU-Pro) of Q8_0
divergences are at word tokens; digits have median margin 12-14 and rarely diverge; the GSM8K
opening token has median margin 0.72.

## 14. Analysis 2: what happens after t\* (`src/classify_divergence.py`)

| | GSM8K Q8_0 | GSM8K Q6_K | GSM8K Q4_K_M | MMLU-Pro Q8_0 | MMLU-Pro Q6_K |
|---|---|---|---|---|---|
| texts re-align within 30 tokens (common block >= 8) | 73% | 73% | 56% | 78% | 71% |
| same final answer | 89% | 86% | 81% | 72% | 64% |
| t\* token class: word / punct / tokenization split / opener | 60 / 20 / 12 / 4% | 61 / 11 / 9 / 16% | 38 / 6 / 4 / 51% | 78 / 11 / 8 / 0% | 81 / 11 / 6 / 1% |

Typical t\* pairs: "So" vs "Therefore", " we" vs " let", " determine" vs " answer", ".\n\n" vs ".",
child ending before " ####". Divergences are mostly local wording edits after which the texts
re-align. Fixing the opener would not help Q4_K_M or Q6_K, which diverge again later.

Offline upper bound for re-anchored exact verification (parent output as draft, verified tokens
cost r = 0.365, unmatched tokens decoded; no verification overhead counted):

| Version | child tokens covered by parent blocks (>= 8) GSM8K / MMLU-Pro | full exact evaluation speedup over decoding |
|---|---|---|
| Q8_0 | 87% / 73% | 2.2x / 1.9x |
| Q6_K | 67% / 51% | 1.7x / 1.5x |
| Q4_K_M | 39% / 33% | 1.3x / 1.3x |

## 15. Net-degradation certificate (`src/simulate_net.py`, `docs/net_certificate.md`)

theta = H - Rep is bounded by an exact upper bound on harms (theorem 4, on A) minus an exact lower
bound on repairs (mirrored binomial inversion, on B = g \ P n D), each at delta'/2. Lemma 3 makes
screening exact on parent-incorrect questions too.

tau = 0.05, delta = 0.05, 9 slices (all + output-length and prompt-length quartiles), |S| = 36,
1,000 reps, r = 0.365. pi floor = minimum inclusion probability on candidates (cover design
otherwise). "cert" = probability that a version passes on all slices simultaneously.
Coverage was >= 0.99 in every cell.

| Dataset | pi floor | Q8_0 cert, harm / net | Q6_K cert, harm / net | cost harm: ours / B1 | cost net: ours / B1 |
|---|---|---|---|---|---|
| GSM8K (1,319) | 0.6 | 0.04 / 0.33 | 0.00 / 0.00 | 0.54 / 0.48 | 0.68 / 0.60 |
| GSM8K (1,319) | 0.8 | 0.52 / 1.00 | 0.00 / 0.15 | 0.67 / 0.63 | 0.85 / 0.80 |
| GSM8K (1,319) | 0.9 | 1.00 / 1.00 | 0.00 / 0.99 | 0.73 / 0.71 | 0.93 / 0.90 |
| GSM8K (1,319) | 0.99 | 1.00 / 1.00 | 0.00 / 1.00 | 0.79 / 0.78 | 1.00 / 0.99 |
| MMLU-Pro (1,000) | 0.8 | 0.00 / 0.00 | 0.00 / 0.00 | 0.38 / 0.32 | 0.94 / 0.80 |
| MMLU-Pro (1,000) | 0.9 | 0.00 / 0.32 | 0.00 / 0.05 | 0.41 / 0.36 | 1.03 / 0.90 |
| MMLU-Pro (1,000) | 0.95 | 0.00 / 0.97 | 0.00 / 0.30 | 0.43 / 0.38 | 1.07 / 0.95 |
| MMLU-Pro (1,000) | 0.99 | 0.00 / 1.00 | 0.00 / 1.00 | 0.45 / 0.40 | 1.11 / 0.99 |

Costs are total audit cost over the 4 versions / full evaluation of the 4 versions. Worst-slice
true (theta, H): GSM8K Q8_0 (0.012, 0.029), Q6_K (0.021, 0.063); MMLU-Pro Q8_0 (0.020, 0.084),
Q6_K (0.028, 0.120).

- The net certificate is the only way to certify GSM8K Q6_K and MMLU-Pro Q8_0 / Q6_K at tau = 0.05:
  their worst-slice harm rate exceeds tau while theta does not.
- It needs inclusion probabilities of 0.9-0.99, i.e. close to full-evaluation cost. In that regime
  exact screening (ours) costs slightly more than B1, because screening all questions is a fixed
  cost and Q4_K_M / Q3_K_M gain nothing from it.
- The zero-harm cover design (pi about 0.33 here) certifies nothing: every version has real flips.

## 16. Re-anchored exact verification, replayed (`src/sim_reanchor.py`)

Replays a prompt-lookup verifier on the known child outputs: the draft is the parent text after
the most recent match of the child's last n tokens; each round verifies L draft tokens in one pass
(cost r (L + 1), accepted prefix + one bonus token), or decodes one token when there is no anchor.
Rejected draft tokens are charged. The output equals the child output by construction. r = 0.365.

| Version | GSM8K: L=4,n=4 / L=16,n=4 / adaptive | MMLU-Pro: L=4,n=4 / L=16,n=4 / adaptive |
|---|---|---|
| Q8_0 | **2.17x** / 1.88x / 2.08x | **1.81x** / 1.47x / 1.74x |
| Q6_K | 1.65x / 1.28x / 1.57x | 1.43x / 1.09x / 1.38x |
| Q4_K_M | 1.26x / 0.93x / 1.26x | 1.23x / 0.93x / 1.23x |
| Q3_K_M | - / 0.81x / - | - / 0.83x / - |

Short drafts win because acceptance is low (Q8_0, L = 16: 76% GSM8K, 61% MMLU-Pro). Caveat:
these numbers assume a verified token costs r = 0.365, measured on long prefills. Short chunks
may use the GPU less efficiently; `src/chunk_cost.py` (job pending) measures the per-token cost of
L-token chunks on a cached prefix. Each round is a sequential step, so latency grows with rounds
(0.19 rounds per token for Q8_0 at L = 16).

## 17. Predicting |A|/|P| from parent margins (`src/predict_D.py`)

sigma_q fitted on the first n_cal questions only; P(in D) = 1 - prod_t Phi(m_t / sigma_q) from
parent margins for the remaining questions. Prediction information only (design, prop. 12).

| n_cal = 50 | GSM8K obs / pred | question AUC | MMLU-Pro obs / pred | question AUC |
|---|---|---|---|---|
| Q8_0 | 0.514 / 0.578 | 0.79 | 0.759 / 0.809 | 0.85 |
| Q6_K | 0.901 / 0.940 | 0.81 | 0.979 / 0.987 | 0.95 |
| Q4_K_M | 0.999 / 0.999 | 0.98 | 1.000 / 1.000 | - |

Fifty child questions plus the parent margins predict |A|/|P| within +0.04 to +0.06, always on
the conservative side; results are stable for n_cal = 100 and 200.

## 18. Flips vs the parent's confidence in its own answer (`src/answer_margin.py`)

Parent answer margin = minimum parent margin over the answer tokens (digits after the last
"####", or the letter after the last "answer is ("), measured along the parent's own trajectory.

| | answer found | answer margin 10 / 50 / 90% | share of harm flips at answer margin < 1 |
|---|---|---|---|
| GSM8K | 1,156 / 1,319 | 2.6 / 17.4 / 20.5 | Q8_0 0%, Q6_K 2%, Q4_K_M 2% |
| MMLU-Pro | 918 / 1,000 | 1.4 / 8.4 / 14.3 | Q8_0 10%, Q6_K 10%, Q4_K_M 5% |

GSM8K Q8_0: all 15 located harm flips have answer margin >= 8 (harm rate 15/820 there, 0/27 below
0.5). Flips are not answer-token near-ties: at the answer position the parent is confident given
its own reasoning, and the child flips because an earlier wording divergence sent it down a
different reasoning path. Answer-position confidence therefore does not predict CoT flips, unlike
decision-margin predictors for single-step decisions.

## 19. MMLU-Pro truncation at Lmax = 1024

The 72 truncated parent outputs are long calculations, not repetition loops: median share of
repeated 20-grams 0.11 (4 of 72 above 0.5; 0.00 for the rest). By category: engineering 25,
chemistry 13, math 11, physics 10, business 7. None reaches "answer is", so parent accuracy on them
is 6.9% (47.5% on the others). Screening is barely affected (Q8_0 share in D 0.82 vs 0.80).
Raising Lmax would add these questions to P with long outputs, which makes screening harder.

## 20. Regime map: when exact screening is cheaper than B1 (`results/regime_map.json`)

Per parent-correct question, screening (lazy, on P) beats B1 at audit rate pi iff
pi > pi* = r (Lp + E[screen prefix]) / (E[full audit] - E[1_D cont audit]).

| Version | pi* at r = 0.2 | r = 0.29 (H100) | r = 0.365 (A100) |
|---|---|---|---|
| GSM8K Q8_0 | 0.32 | 0.46 | 0.59 |
| GSM8K Q6_K | 0.53 | 0.82 | never |
| MMLU-Pro Q8_0 | 0.47 | 0.69 | 0.88 |
| GSM8K Q4_K_M, MMLU-Pro Q6_K / Q4_K_M | never | never | never |

Uniform sufficient audit rate of the cover design, 1 - (delta/|S|)^(1/k), delta = 0.05:

| N_g | tau | \|S\| = 36 | 200 | 2000 |
|---|---|---|---|---|
| 100 | 0.05 | 0.73 | 0.81 | 0.88 |
| 300 | 0.05 | 0.36 | 0.42 | 0.51 |
| 300 | 0.02 | 0.67 | 0.75 | 0.83 |
| 1000 | 0.05 | 0.12 | 0.15 | 0.19 |
| 1000 | 0.02 | 0.28 | 0.34 | 0.41 |
| 1000 | 0.01 | 0.48 | 0.56 | 0.65 |
| 3000 | 0.01 | 0.20 | 0.24 | 0.30 |

Exact screening pays off only when three things hold together: a high bit-width version (Q8_0),
fine-grained simultaneous certification (slices of about 100-300 or tau <= 0.02), and a low
prefill/decode cost ratio r.

## 21. Measured cost of short verification chunks (`src/chunk_cost.py`, A100, fp32 Q8_0)

Cost only, prefix caching on (never used for truth data). 512 GSM8K questions; each request
extends a fully cached prefix by L tokens; per-token time over decode per-token time:

| L | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| r(L) | 2.01 | 1.19 | 0.83 | 0.58 | 0.48 | 0.42 | 0.40 |

Short chunks are far more expensive per token than a long prefill (0.365): the fixed cost of a
forward pass is not amortised. Replaying the re-anchored verifier (sec. 16, n = 4) with r(L):

| Version | GSM8K L=4 / 8 / 16 / 32 | MMLU-Pro L=4 / 8 / 16 / 32 |
|---|---|---|
| Q8_0 | 1.11 / 1.43 / **1.52** / 1.39x | 1.05 / **1.25** / 1.23 / 1.04x |
| Q6_K | 1.01 / **1.15** / 1.08 / 0.88x | 0.96 / **1.03** / 0.93 / 0.73x |
| Q4_K_M | 0.91 / 0.93 / 0.81 / 0.62x | 0.92 / 0.93 / 0.82 / 0.63x |

Exact full evaluation by re-anchored verification therefore gains at most about 1.5x (Q8_0,
GSM8K) and nothing below Q6_K. These chunk costs come from separate lock-step generate calls; an
engine that mixes verification and decoding in one batch could do better, so treat them as
conservative.

## 22. Second parent (exploratory, DECISIONS.md D8): Phi-3.5-mini-instruct

Ladder dequantization works for phi3 (median / max relative weight error: Q8_0 0.54% / 0.77%,
Q6_K 1.78% / 2.34%, Q4_K_M 2.34% / 7.5%, Q3_K_M 7.2% / 15.1%; BPW 8.50 / 6.56 / 5.02 / 4.11).

**Determinism check fails in the same fp32 batch-invariant config that Qwen passes**
(200 GSM8K questions, A100, job 2276157):

| Layer | Phi-3.5-mini | Qwen2.5-3B |
|---|---|---|
| 1 repeat run, separate processes | **5/200 differ** (1 answer) | 0 |
| 2 prefill vs decode | 5 tokens in the same 5 questions | 0 |
| 3 continuation | 5/400 | 0 |
| 4 batch 256 -> 8, shuffled | 1/200 (a different question) | 0 |

Processes A and C agree on those 5 questions and only process B differs (its prefill also
disagrees with A), so the variation is between processes, not with batch composition. A likely
cause is kernel selection that depends on timing (autotuning) for matmul shapes outside vLLM's
tuned batch-invariant table (Phi-3.5: hidden 3072, intermediate 8192; Qwen2.5-3B's shapes are in
it). This is the plan's E4 risk (Vosti 2026) showing up for a second model family:
batch-invariant mode alone does not guarantee the prefill-decode invariance of definition 2.

**With `--enforce-eager` (no torch.compile, no CUDA graphs) all four layers pass** (job 2276333):
three separate processes produce identical outputs (52,315 tokens each), prefill top-1 matches
decoding on every token, all 400 continuations match, batch 256 -> 8 changes nothing. Decode is
about as fast as compiled mode here (gen A 37.5 s vs 32.8 s). So the cross-process variation comes
from the compiled path, and eager execution is the deterministic configuration for Phi-3.5.
The execution configuration of every truth matrix must therefore record the compile mode, and
the four-layer check has to be rerun per model family rather than assumed from another family.

## 23. Plan metric (iv): worst-slice bound at equal compute (`src/eq_budget.py`)

Per version, the inclusion-probability floor is bisected so the expected audit cost equals a
budget b (fraction of that version's full evaluation); value = mean worst-slice UCB over 400
reps, tau = 0.05 design, 9 slices, |S| = 36, r = 0.365. "-" = the minimum cost of the method
(screening, or the cover design itself) exceeds b.

GSM8K, true worst-slice H / theta: Q8_0 0.029 / 0.012, Q6_K 0.063 / 0.021.

| Version, method | b = 0.4 | 0.6 | 0.8 | 1.0 |
|---|---|---|---|---|
| Q8_0 harm, B1 | 0.078 | 0.053 | 0.029 | 0.029 |
| Q8_0 harm, exact screening | 0.094 | **0.032** | 0.029 | 0.029 |
| Q8_0 net, B1 | 0.089 | 0.056 | 0.037 | 0.012 |
| Q8_0 net, exact screening | - | 0.056 | **0.012** | 0.012 |
| Q6_K harm, B1 | 0.127 | 0.097 | 0.063 | 0.063 |
| Q6_K harm, exact screening | 0.158 | 0.105 | 0.071 | 0.063 |
| Q6_K net, B1 | 0.133 | 0.088 | 0.056 | 0.021 |
| Q6_K net, exact screening | - | 0.112 | 0.068 | 0.033 |

MMLU-Pro (true worst-slice H / theta: Q8_0 0.084 / 0.020, Q6_K 0.120 / 0.028): screening is
never better; harm bounds saturate at H by b = 0.4-0.6, and the net bound with screening trails B1
(Q8_0 at b = 0.8: 0.086 vs 0.073).

At b = 0.6 on GSM8K Q8_0, the screened harm certificate is nearly exact (width 0.003 vs 0.024 for
B1, about 8x tighter); at b = 0.8 the screened net certificate reaches the exact theta (0.012 vs
0.037). This is the regime of the break-even analysis (sec. 20, pi* = 0.59): exact screening
wins at equal compute only for the high bit-width version, on the shorter-output task, and at
budgets around 0.6-0.8 of full evaluation.

## 24. Phi-3.5-mini truth matrix (GSM8K 1,319, fp32, eager, A100; job 2276333)

| Version | Child acc. | \|A\|/\|P\| (Qwen) | median t\* | harm / repair | lemma 3 violations | ties |
|---|---|---|---|---|---|---|
| parent | 82.0% | 0 | - | - | 0 | 0 |
| Q8_0 | 80.7% | **0.414** (0.519) | 92 | 30 / 13 | 0 | 1 question |
| Q6_K | 81.5% | **0.779** (0.902) | 53 | 38 / 32 | 0 | 0 |
| Q4_K_M | 77.5% | 0.974 (0.999) | 20 | 113 / 54 | 0 | 0 |
| Q3_K_M | 73.8% | 1.000 (1.000) | 13 | 152 / 45 | 0 | 0 |

Local decoding of Phi tokens matches vLLM text on 1,319/1,319 outputs.

**The margin scaling law holds across families** (rho/sigma vs f_m(0)/sqrt(2 pi)):

| Parent / dataset | f_m(0) | predicted rho/sigma | observed rho/sigma, Q8 / Q6 / Q4 / Q3 |
|---|---|---|---|
| Qwen2.5-3B / GSM8K | 0.12 | 0.048 | 0.040 / 0.044 / 0.037 / (not noise-like) |
| Qwen2.5-3B / MMLU-Pro | 0.19 | 0.077 | 0.071 / 0.074 / 0.068 / - |
| Phi-3.5-mini / GSM8K | 0.07 | 0.028 | 0.027 / 0.027 / 0.023 / 0.026 |

Noise sd per bit width is nearly family-independent (Phi 0.08 / 0.26 / 0.90 vs Qwen 0.07 / 0.23 /
0.83 for Q8 / Q6 / Q4). Phi has 60% of Qwen's parent near-tie density, which is why its |A|/|P| is
lower: screening power is mostly a property of the parent's own indecision. The Gaussian model
is also calibrated at the question level for Phi (Q8_0 |D|/N pred 0.446, obs 0.432).

Other replications: divergences are lexical (word 71%, punctuation 15% for Q8_0), 78% re-align
within 30 tokens, 88% keep the answer; harm flips occur where the parent is confident at the answer
position (0 of 23 located Q8_0 harms below answer margin 1). Flips are **not** symmetric for Phi
(harm / repair 30/13 Q8_0, 152/45 Q3_K_M): quantization degrades Phi systematically, unlike the
near-symmetric flips of Qwen.

Equal budget (sec. 23 method), Phi Q8_0, true worst-slice H / theta 0.051 / 0.038:

| Method | b = 0.6 | b = 0.8 |
|---|---|---|
| harm: screening / B1 | **0.051** / 0.082 | 0.051 / 0.051 |
| net: screening / B1 | 0.078 / 0.092 | **0.038** / 0.069 |

The Q8_0 equal-budget advantage replicates: screened certificates reach the exact H at b = 0.6
and the exact theta at b = 0.8. For Q6_K the two are close (net, b = 0.8: 0.045 vs 0.048).
Predictor (sec. 17) with 50 calibration questions: Q8_0 overestimates (0.553 vs 0.414; few
divergences in 50 questions), Q6_K 0.796 vs 0.783, Q4_K_M 0.975 vs 0.976.

## 25. MMLU-Pro 3,000 (14 categories x about 214), Qwen2.5-3B, fp32, A100 (job 2276120)

| Version | Child acc. | \|A\|/\|P\| (pilot 1,000) | harm / repair | lemma 3 violations | flips outside D | questions with ties |
|---|---|---|---|---|---|---|
| Q8_0 | 44.8% | **0.766** (0.756) | 138 / 160 | 0 | 0 | 0 |
| Q6_K | 44.3% | 0.975 (0.980) | 241 / 246 | 0 | 0 | 1 |
| Q4_K_M | 42.1% | 1.000 | 324 / 263 | 0 | 0 | 0 |
| Q3_K_M | 23.7% | 1.000 | 832 / 221 | 0 | 0 | 2 |

Parent accuracy 44.1%, |P| = 1,323. The pilot numbers replicate; Qwen's high-bit flips are
symmetric or slightly repair-heavy here.

Certificates with all + length/prompt quartiles + 14 category slices (23 slices, m = 100,
|S| = 92), tau = 0.05, r = 0.365:

| | ours | sel | B1 |
|---|---|---|---|
| speedup over full evaluation of the ladder | 3.76 | 4.74 | 5.10 |
| simultaneous coverage | 0.971 | 0.975 | 0.979 |
| Q8_0 worst-slice UCB (cover design) | 0.171 | 0.171 | 0.173 |

True worst-slice theta / H: Q8_0 0.028 / 0.079, Q6_K 0.033 / 0.112. With an inclusion floor of
0.95 the net certificate passes Q8_0 on all 23 slices in 37% of reps (harm certificate: never);
cost 1.08 (ours) vs 0.95 (B1) of full evaluation.

## 26. Third family (exploratory): SmolLM2-1.7B-Instruct @ 31b70e2, GSM8K 1,319, fp32, eager

Apache-2.0, not gated, llama architecture, single-file checkpoint. Four-layer determinism check
passes in eager mode (job 2276609). Weight error (median): Q8_0 0.54%, Q6_K 1.79%, Q4_K_M 7.17%,
Q3_K_M 7.27%. Local decoding matches vLLM text on 1,319/1,319 after treating vocab-listed special
tokens as empty (`src/tokdecode.py`; Phi and Qwen outputs unchanged, checked).

| Version | \|A\|/\|P\| | harm / repair | same answer in D | lemma 3 violations |
|---|---|---|---|---|
| Q8_0 | **0.470** | 42 / 58 | 57% | 0 |
| Q6_K | 0.878 | 105 / 115 | 52% | 0 |
| Q4_K_M | 0.996 | 197 / 136 | 37% | 0 |
| Q3_K_M | 1.000 | 333 / 92 | 23% | 0 |

Parent accuracy 41.2% (|P| = 543). The model rarely writes "####" (answer span found in 230/1,319),
so the answer-margin analysis is not informative here.

**Margin scaling law, all four parent x task combinations:**

| Parent / dataset | f_m(0) | predicted rho/sigma | observed rho/sigma, Q8 / Q6 / Q4 / Q3 |
|---|---|---|---|
| Qwen2.5-3B / GSM8K | 0.12 | 0.048 | 0.040 / 0.044 / 0.037 / (not noise-like) |
| Qwen2.5-3B / MMLU-Pro | 0.19 | 0.077 | 0.071 / 0.074 / 0.068 / - |
| Phi-3.5-mini / GSM8K | 0.07 | 0.028 | 0.027 / 0.027 / 0.023 / 0.026 |
| SmolLM2-1.7B / GSM8K | 0.124 | 0.049 | 0.047 / 0.048 / 0.050 / 0.053 |

Noise sd per bit width (Q8 / Q6 / Q4): Qwen 0.07 / 0.23 / 0.83, Phi 0.08 / 0.26 / 0.90,
SmolLM2 0.076 / 0.26 / 0.77. SmolLM2's Q4_K_M has 2.5x the median weight error of the other two
families but the same logit noise, so median weight error is a poor proxy for sigma_q.
Predictor with 50 calibration questions: Q8_0 0.466 vs 0.467 observed, Q6_K 0.901 vs 0.879.
Flip direction is model-dependent: SmolLM2 Q8_0 is repair-heavy (42/58), Phi harm-heavy, Qwen even.

Equal budget, SmolLM2 Q8_0 (true worst-slice H / theta 0.052 / 0.003): net certificate with
screening 0.009 at b = 0.8 vs 0.040 for B1 (about 4.4x tighter); harm certificates saturate at H
by b = 0.4 for both. Q6_K: screening never better.

## 27. Phi-3.5-mini on MMLU-Pro pilot (1,000), fp32, eager (job 2276607)

| Version | Child acc. | \|A\|/\|P\| (Qwen) | harm / repair | lemma 3 violations | flips outside D |
|---|---|---|---|---|---|
| Q8_0 | 52.0% | **0.628** (0.756) | 40 / 44 | 0 | 0 |
| Q6_K | 50.4% | 0.919 (0.980) | 83 / 71 | 0 | 0 |
| Q4_K_M | 48.8% | 0.990 (1.000) | 110 / 82 | 0 | 0 |
| Q3_K_M | 44.8% | 0.996 (1.000) | 141 / 73 | 0 | 0 |

Parent accuracy 51.6%. Decoding matches vLLM text on 1,000/1,000.

Margin law, fifth parent x task pair: f_m(0) = 0.164, predicted rho/sigma 0.065, observed
0.058 / 0.063 / 0.055 / 0.059 (Q8 / Q6 / Q4 / Q3); sigma 0.073 / 0.24 / 0.75 / 1.1.
Predictor (50 calibration questions): Q8_0 0.674 vs 0.634, Q6_K 0.928 vs 0.920.

| Parent / dataset | f_m(0) | predicted rho/sigma | observed rho/sigma, Q8 / Q6 / Q4 / Q3 |
|---|---|---|---|
| Qwen2.5-3B / GSM8K | 0.12 | 0.048 | 0.040 / 0.044 / 0.037 / - |
| Qwen2.5-3B / MMLU-Pro | 0.19 | 0.077 | 0.071 / 0.074 / 0.068 / - |
| Phi-3.5-mini / GSM8K | 0.07 | 0.028 | 0.027 / 0.027 / 0.023 / 0.026 |
| Phi-3.5-mini / MMLU-Pro | 0.164 | 0.065 | 0.058 / 0.063 / 0.055 / 0.059 |
| SmolLM2-1.7B / GSM8K | 0.124 | 0.049 | 0.047 / 0.048 / 0.050 / 0.053 |

Flip symmetry depends on the task, not only the model: the same Phi Q8_0 is harm-heavy on GSM8K
(30/13) and even on MMLU-Pro (40/44). On MMLU-Pro, exact screening is again never better at
equal budget (Q8_0 net at b = 0.8: 0.088 vs 0.066 for B1), as for Qwen.

## 28. Robustness of the margin law (`src/law_robustness.py`)

ratio = observed rho / (f_m(0) sigma / sqrt(2 pi)); sigma = sd of d over positions with parent
margin < 4; f_m(0) = share of parent margins below w, divided by w. 95% bootstrap intervals
over questions (200 resamples). Values for w = 0.1:

| Parent / dataset | Q8_0 | Q6_K | Q4_K_M | Q3_K_M |
|---|---|---|---|---|
| Qwen2.5-3B / GSM8K | 0.81 [0.77, 0.85] | 0.88 [0.85, 0.91] | 0.74 [0.71, 0.76] | 1.12 [1.08, 1.15] |
| Qwen2.5-3B / MMLU-Pro | 0.89 [0.85, 0.93] | 0.91 [0.89, 0.93] | 0.83 [0.81, 0.85] | 1.05 [1.02, 1.08] |
| Phi-3.5-mini / GSM8K | 0.91 [0.86, 0.97] | 0.94 [0.90, 0.99] | 0.80 [0.77, 0.84] | 0.90 [0.86, 0.93] |
| Phi-3.5-mini / MMLU-Pro | 0.83 [0.79, 0.87] | 0.90 [0.87, 0.93] | 0.81 [0.79, 0.83] | 0.87 [0.85, 0.90] |
| SmolLM2-1.7B / GSM8K | 0.94 [0.89, 0.98] | 0.92 [0.89, 0.95] | 0.98 [0.95, 1.02] | 1.06 [1.03, 1.09] |
| Qwen2.5-14B / GSM8K (sec. 33) | 0.87 [0.81, 0.94] | 0.91 [0.88, 0.95] | 0.85 [0.82, 0.89] | 0.88 [0.86, 0.91] |
| Llama-3.1-8B / GSM8K (sec. 34) | 0.89 [0.83, 0.94] | 0.83 [0.80, 0.86] | 0.80 [0.77, 0.82] | 0.85 [0.83, 0.88] |

All 20 ratios lie in 0.72-1.14 (median about 0.90). The window w (0.05, 0.1, 0.2) changes them by
at most 0.03. The intervals are narrow and most ratios are below 1, so the Gaussian law
over-predicts by about 10% systematically (most for Q4_K_M), consistent with a lighter-than-Gaussian
perturbation tail near zero margin. As a first-order description, rho ~ c f_m(0) sigma / sqrt(2 pi)
with c ~ 0.9.

## 29. Prefill/decode cost at 7B (cost only; Qwen2.5-7B-Instruct parent, fp32, compiled, A100)

Same 1,319 GSM8K prefill sequences as sec. 9 (the 3B parent outputs; the two models share the
tokenizer), job 2276787.

| Model | decode s/token | prefill (top-2) s/token | r | bare-prefill r |
|---|---|---|---|---|
| Qwen2.5-3B Q8_0 | 2.45e-4 | 0.89e-4 | 0.365 | 0.308 |
| Qwen2.5-7B parent | 4.01e-4 (1.6x) | 1.87e-4 (2.1x) | **0.467** | 0.40 |

The cost ratio gets worse with scale in this regime: batched (256) prefill is compute-bound and
grows with parameter count, while batched decode is partly bandwidth-bound and grows less. In
batched offline evaluation, larger models make exact screening relatively more expensive, the
opposite of the small-batch serving intuition behind E3.

## 30. Decode batch size decides the economics (cost only; Qwen2.5-3B Q8_0, fp32, A100)

Same 1,319 prefill sequences, max_num_seqs = 32 instead of 256 (job 2276900):

| max_num_seqs | decode s/token | prefill (top-2) s/token | r |
|---|---|---|---|
| 256 | 2.45e-4 | 0.90e-4 | 0.365 |
| 32 | 11.8e-4 (4.8x) | 0.90e-4 (same) | **0.076** |

At small decode batches decoding is bandwidth-bound and per-token cost rises about 5x, while prefill
throughput does not depend on the decode batch. Break-even audit rate pi* at r = 0.076
(sec. 20 formula):

| Parent / dataset | Q8_0 | Q6_K | Q4_K_M |
|---|---|---|---|
| Qwen2.5-3B / GSM8K | 0.12 | 0.19 | 0.74 |
| Qwen2.5-3B / MMLU-Pro | 0.17 | 0.41 | never |
| Phi-3.5-mini / GSM8K | 0.11 | 0.14 | 0.24 |
| SmolLM2-1.7B / GSM8K | 0.13 | 0.19 | 0.67 |

Qwen GSM8K at r = 0.076, cover design (tau 0.05, 9 slices): Q8_0 audit cost 57k vs 97k for B1
(0.59x), ladder speedup 4.31 (ours) / 4.34 (sel) vs 3.90 (B1). Equal budget (worst-slice UCB, true
H / theta 0.029 / 0.012):

| Q8_0 | b = 0.4 | b = 0.6 |
|---|---|---|
| harm: screening / B1 | **0.029** (exact) / 0.078 | 0.029 / 0.054 |
| net: screening / B1 | 0.030 / 0.089 | **0.012** (exact) / 0.056 |

Q6_K also gains (net at b = 0.8: 0.021 exact vs 0.056). Which regime applies is set by the
largest decode batch that fits in memory: 3B and 7B at 80 GB fit 256 sequences (unfavourable,
r = 0.37-0.47), while larger models, longer outputs or less memory force small batches, where exact
screening gives several-fold savings.

## 31. Scale: Qwen2.5-7B-Instruct truth matrix (GSM8K 1,319, fp32, eager; job 2276789)

Four-layer check passes in eager mode. The job ended with exit code 6 from a Python finalization
crash in the last engine teardown, after all outputs (5 screens, 5 margin profiles) were written.
Decoding matches vLLM text on 1,319/1,319. Dequantization error (median): Q8_0 0.54%, Q6_K 1.80%,
Q4_K_M 2.15%, Q3_K_M 7.22%.

| Version | Child acc. | \|A\|/\|P\| (3B) | harm / repair | sigma (3B) | lemma 3 violations |
|---|---|---|---|---|---|
| Q8_0 | 87.7% | **0.317** (0.519) | 13 / 6 | 0.055 (0.074) | 0 |
| Q6_K | 87.9% | **0.691** (0.902) | 27 / 23 | 0.17 (0.24) | 0 |
| Q4_K_M | 87.5% | 0.947 (0.999) | 46 / 36 | 0.53 (0.86) | 0 |
| Q3_K_M | 86.5% | 0.994 (1.000) | 68 / 45 | 1.0 (2.8) | 0 |

Parent accuracy 88.2%. Two effects make the larger model screen better: fewer parent near-ties
(f_m(0) = 0.078 vs 0.12 for 3B) and 25-40% smaller quantization noise per bit width. So sigma_q is
family-independent at a given size but shrinks with scale. Margin law, sixth pair: observed /
predicted 0.92 / 0.91 / 0.91 / 0.94 (Q8 / Q6 / Q4 / Q3). Predictor (50 questions): Q8_0 0.314 vs
0.322 observed.

Equal budget, Q8_0 (true worst-slice H / theta 0.026 / 0.023), harm certificate at b = 0.6:
screening 0.026 (exact) vs B1 0.054 at r = 0.365; 0.045 vs 0.054 at r = 0.467 (7B, batch 256);
at r = 0.076 the screened certificate is already exact at b = 0.4 (B1 0.075). Q6_K at r = 0.076:
net certificate exact at b = 0.6 (0.023 vs 0.074).

## 32. Cost ratio vs decode batch (Qwen2.5-3B Q8_0, fp32, A100; jobs 2277107, 2276900, 2277108, 2277109)

Same 1,319 prefill sequences and decodes; only max_num_seqs changes.

| max_num_seqs | decode time | decode s/token | prefill (top-2) time | r |
|---|---|---|---|---|
| 16 | 886.7 s | 24.0e-4 | 47.1 s | 0.039 |
| 32 | 437.8 s | 11.8e-4 | 45.9 s | 0.076 |
| 64 | 235.9 s | 6.38e-4 | 45.4 s | 0.140 |
| 128 | 122.8 s | 3.32e-4 | 45.6 s | 0.269 |
| 256 | 90.5 s | 2.45e-4 | 45.6 s | 0.365 |

Decode time halves with each doubling of the batch up to 128 (bandwidth-bound) and improves only
26% from 128 to 256 (approaching compute-bound); prefill time does not depend on the decode batch.
From 16 to 128 sequences, r is roughly proportional to the batch (0.0019-0.0024 per sequence).

Correction (2026-10-31, `src/concurrency_sim.py`): replaying the scheduler for these runs gives a
nearly constant time per decode step, 37.8 / 37.0 / 38.7 / 35.6 / 42.0 ms at caps 16 / 32 / 64 /
128 / 256 (mean batches 16 / 31 / 61 / 108 / 172). Reading 12.3 GB of fp32 weights in 37 ms is
about 330 GB/s, a sixth of the A100's HBM bandwidth, so the decode step is not HBM-bound: its cost
is a fixed per-step cost of the batch-invariant engine. The "bandwidth-bound" wording above and in
sec. 30 should be read as "fixed per-step cost". The conclusion is unchanged and gets a simpler form:
r ~ B_eff x (prefill seconds per token) / (seconds per decode step), with B_eff the mean decode batch.

## 33. Qwen2.5-14B-Instruct truth matrix (GSM8K 1,319, fp32, eager, A100; job 2280619)

Parent @ cf98f3b (Apache-2.0). Four-layer check passes in eager mode (L1-L4: 0 differences on 200
questions). In fp32 the weights take about 59 GB, leaving a KV cache of 27,760 tokens (about 6.8
sequences of 4,096). Decoding matches vLLM text on 1,319/1,319. Dequantization error (median per
tensor): Q8_0 0.54%, Q6_K 1.79%, Q4_K_M 2.84%, Q3_K_M 7.17%.

| Version | Child acc. | \|A\|/\|P\| (7B / 3B) | harm / repair | sigma (7B) | lemma 3 violations | t\* median |
|---|---|---|---|---|---|---|
| Q8_0 | 91.7% | **0.398** (0.317 / 0.519) | 8 / 6 | 0.053 (0.055) | 0 | 71 |
| Q6_K | 91.3% | 0.800 (0.691 / 0.902) | 18 / 10 | 0.167 (0.17) | 0 | 41 |
| Q4_K_M | 92.0% | 0.977 (0.947 / 0.999) | 24 / 26 | 0.45 (0.53) | 0 | 17 |
| Q3_K_M | 90.1% | 0.998 (0.994 / 1.000) | 47 / 23 | 0.89 (1.0) | 0 | 10 |

Parent accuracy 91.9%. No flips outside D, no ties. The 14B parent screens worse than 7B, so
"larger screens better" (sec. 31) was too coarse: sigma_q stops shrinking between 7B and 14B (Q8_0
0.055 -> 0.053), while the 14B parent has more near-ties (share of positions with margin < 0.1:
1.05%, f_m(0) about 0.105, vs 0.078 for 7B) at the same output length (271 vs 270 tokens). The
margin law accounts for the difference: observed / predicted 0.87 / 0.91 / 0.85 / 0.88 (Q8 / Q6 /
Q4 / Q3; w = 0.1). Screening power follows the parent's near-tie density, not parameter count.
Predictor (50 questions): Q8_0 0.414 vs 0.399, Q6_K 0.781 vs 0.802.

Divergences: 72-83% word choices; 80% re-sync within 30 tokens (Q8_0); 97% of Q8_0 divergences keep
the answer. Harms come with high parent answer margins (7 of the 8 Q8_0 harms at answer margin >= 8;
the 8th output has no detected answer span), i.e. from upstream drift (as sec. 18).

Measured cost in the truth run (eager, KV-limited): r = 0.31 (decode 437 s, top-2 prefill 189 s for
Q8_0). Replaying the scheduler (`src/concurrency_sim.py`) gives a mean decode batch of about 100 and
about 117 ms per step: the 28k-token KV cache still holds about 100 short GSM8K sequences, so this is
not a small-batch regime. Equal budget at r = 0.365, Q8_0 (true worst-slice H / theta 0.015 /
0.012): harm certificate at b = 0.6 is 0.028 (screening) vs 0.043 (B1), at b = 0.8 exact (0.015)
vs 0.028; net at b = 0.8 exact (0.012) vs 0.032. At r = 0.076 screening is exact at b = 0.4 for both
(B1 0.063 / 0.073). Q6_K gains only at r = 0.076 (harm 0.045 vs 0.069 at b = 0.6).

## 34. Fourth family: Llama-3.1-8B-Instruct truth matrix (GSM8K 1,319, fp32, eager, A100; job 2280621)

Parent @ 0e9e39f (Llama 3.1 Community License; gated, access granted to the user's account).
Four-layer check passes in eager mode. KV cache 145,232 tokens, so the decode batch reaches the cap of
256. Decoding matches vLLM text on 1,319/1,319. Dequantization error (median): Q8_0 0.54%, Q6_K 1.80%,
Q4_K_M 7.18%, Q3_K_M 7.30% (Q4_K_M error is much higher than for Qwen, 2-3%).

| Version | Child acc. | \|A\|/\|P\| | harm / repair | sigma | lemma 3 violations | t\* median |
|---|---|---|---|---|---|---|
| Q8_0 | 82.9% | **0.420** | 22 / 24 | 0.050 | 0 | 60 |
| Q6_K | 83.6% | 0.788 | 38 / 49 | 0.170 | 0 | 38 |
| Q4_K_M | 82.3% | 0.983 | 70 / 64 | 0.48 | 0 | 23 |
| Q3_K_M | 80.7% | 0.998 | 91 / 63 | 0.86 | 0 | 14 |

Parent accuracy 82.8%. One parent question has an exact tie (resolved by the smallest-id rule, no
violations); no flips outside D. f_m(0) about 0.137 (share < 0.1: 1.37%), the highest of all GSM8K
parents, which explains |A|/|P| despite the smallest sigma. Margin law: observed / predicted 0.89 /
0.83 / 0.80 / 0.85. Flips are close to symmetric and repair-heavy for Q6_K (38 / 49), so the
harm certificate is loose and the net certificate is the useful one: at r = 0.076, Q8_0 net bound is
exact (0.006) at b = 0.6 vs 0.058 for B1, and 0.020 at b = 0.4 vs 0.094; Q6_K net is exact (0.000) at
b = 0.8 vs 0.043. At r = 0.365 (the truth run measured r = 0.36 at batch 256) screening wins only
for Q8_0 at b >= 0.6. 85% of Q8_0 divergences keep the answer (Qwen-14B 97%); Llama's first
divergences include list formatting ("Step" | "1").

With these two parents the margin law covers 32 parent x task x version cases; all ratios lie in
0.72-1.14 and the median is about 0.89 (`results/law_robustness_q14_l8.json`).

## 35. Baselines B2 and B6 at equal compute (`src/baselines.py`)

Harm certificate, worst slice over the 9 GSM8K slices (or MMLU-Pro length slices), simultaneous over
4 versions x slices (delta' = 1.4e-3). Each method spends the same expected compute b (fraction of a
full evaluation of that version); B1 and ours as in sec. 23, the new methods with uniform Poisson
sampling on their candidate set.

- **B6** (common practice): Horvitz-Thompson estimate on g n P plus a CLT interval with Bonferroni.
  Asymptotic, so its coverage is reported.
- **B2** (PPI / AIPW): AIPW on g n P with a predictor from parent signals only, f_i = q x P(i in D).
  P(i in D) comes from the margin law with sigma fitted on 50 questions (sec. 17); q is the
  flip-given-divergence rate transferred from another parent's truth matrix. Finite-sample bound from
  Bernstein's inequality (variance bounded by T + sum f^2, solved in closed form).
- **B2s**: the same AIPW bound on the screened set A = g n P n D with f_i = q, paying the prefill.
  This is PPI on top of exact screening.

Q8_0, mean worst-slice UCB (B6 coverage in brackets), r = 0.076:

| Parent / task | true H | b = 0.4: B1 / ours / B2 / B2s / B6 | b = 0.6: B1 / ours / B2 / B2s / B6 |
|---|---|---|---|
| Qwen-3B GSM8K | 0.029 | 0.078 / **0.029** / 0.101 / 0.029 / 0.061 [0.30] | 0.054 / **0.029** / 0.066 / 0.029 / 0.046 [0.74] |
| Qwen-7B GSM8K | 0.026 | 0.075 / **0.026** / 0.101 / 0.026 / 0.055 [0.12] | 0.054 / **0.026** / 0.066 / 0.026 / 0.043 [0.32] |
| Qwen-14B GSM8K | 0.015 | 0.063 / **0.015** / 0.093 / 0.015 / 0.042 [0.29] | 0.043 / **0.015** / 0.058 / 0.015 / 0.031 [0.60] |
| Llama-8B GSM8K | 0.030 | 0.078 / **0.030** / 0.101 / 0.030 / 0.062 [0.41] | 0.054 / **0.030** / 0.065 / 0.030 / 0.046 [0.72] |
| Phi-3.5 GSM8K | 0.051 | 0.108 / **0.051** / 0.133 / 0.051 / 0.092 [0.67] | 0.082 / **0.051** / 0.095 / 0.051 / 0.074 [0.85] |
| Qwen-3B MMLU-Pro | 0.084 | 0.092 / **0.084** / 0.103 / 0.084 / 0.088 [0.75] | all 0.084 |

At r = 0.365 and b = 0.6, Q8_0: B1 / ours = 0.053 / 0.032 (Qwen-3B), 0.054 / 0.026 (7B), 0.043 / 0.028
(14B), 0.054 / 0.030 (Llama), 0.082 / 0.051 (Phi). At b = 0.4 and r = 0.365, B1 is better than ours
everywhere except Qwen-3B MMLU-Pro (0.084 vs 0.091). For Qwen-14B the prefill alone exceeds the
budget.

What this shows:
1. **B2 is looser than B1 in every case**: 0.093-0.133 vs 0.063-0.108 at b = 0.4. A predictor from
   parent signals carries little information about harm (flips are about 3-7% of divergent
   questions). The finite-sample AIPW bound must pay for the 1/pi range of its terms, which costs more
   than the exact binomial inversion of theorem 4. B2s never beats ours (equal at best). This matches
   the plan's expectation that B2-B5 cannot beat the lower bound set by the candidate-set size for
   worst-slice certificates; only exact information shrinks the candidate set.
2. **B6 is narrower than B1 only because it is invalid.** Its simultaneous coverage is 0.12-0.74 at
   b = 0.4-0.6 for Q8_0 (target 0.95). The CLT interval collapses to 0 when a slice's sample has no
   flips. Even so, ours is narrower than B6 whenever the screened certificate is exact.
3. **G3 (>= 1.5x over the strongest valid baseline, B1).** UCB ratio B1 / ours for Q8_0:
   - At r = 0.076, b = 0.4: 2.7 (Qwen-3B), 2.9 (7B), 4.2 (14B), 2.6 (Llama), 2.1 (Phi).
   - At r = 0.365, b = 0.6: 1.66, 2.08, 1.54, 1.80, 1.61.
   - Q6_K at r = 0.076, b = 0.6: 1.28 (Qwen-3B), 1.70 (7B), 1.53 (14B), 1.48 (Llama), 1.62 (Phi).
4. SmolLM2 and some MMLU-Pro cells are equal for every method. Parent accuracy is low there, so the
   harm candidates (parent-correct questions) can all be audited within the budget.

Not simulated: B3 (active inference), B4 (cost-optimal active evaluation) and B5 (FAQ / PAI). They
change only the sampling design. Under note 2 of the plan, tilting inclusion probabilities can save
at most k - 1 audits in a worst-case slice certificate. B2 uses Bernstein; a betting-based
finite-population confidence sequence would be tighter, but it remains bounded by the same
candidate-set argument.

## 36. What sets r: a fixed cost per decode step (cost only; jobs 2281338-2281341)

Per-token r = (top-2 prefill s/token) / (decode s/token). The time per step comes from replaying the
scheduler (`src/concurrency_sim.py`) on the measured decode wall time.

| Model (mode) | questions | cap | mean decode batch | ms / step | r |
|---|---|---|---|---|---|
| Qwen-3B Q8_0 (compiled) | 1,319 | 16 / 32 / 64 / 128 / 256 | 16 / 31 / 61 / 108 / 172 | 37.8 / 37.0 / 38.7 / 35.6 / 42.0 | 0.038 / 0.076 / 0.140 / 0.269 / 0.365 |
| Qwen-3B Q8_0 (eager) | 1,319 | 32 | 31 | 39.0 | 0.072 |
| Qwen-14B parent (eager) | 200 | 16 / 64 / 256 | 15 / 48 / 73 (KV-limited, 310 preemptions) | 92 / 94 / 113 | 0.062 / 0.189 / 0.216 |
| Qwen-14B parent (eager, truth run) | 1,319 | 256 | about 100 (KV-limited) | 117 | 0.31 |
| Llama-8B parent (eager, truth run) | 1,319 | 256 | 171 | 87 | 0.36 |

1. The time per decode step hardly depends on the batch (3B: 36-42 ms from 16 to 256 sequences;
   14B: 92-113 ms). Compiled mode (CUDA graphs) saves only 6% over eager, so the step cost is GPU
   kernel time, not CPU launch overhead. 3B reads its 12.3 GB of fp32 weights in about 37 ms (about
   330 GB/s, a sixth of HBM bandwidth), so the step is bound by the batch-invariant matmul kernels at
   small M. It is neither HBM-bound nor compute-bound.
2. Hence **r ~ kappa x B_eff**, with kappa = (prefill s/token) / (s/step): 0.0024 for Qwen-3B and
   0.0040 for Qwen-14B (prefill cost grows with parameters faster than step cost). This predicts
   14B at batch 16 as 0.064 (measured 0.062).
3. B_eff = min(cap, KV tokens / mean sequence occupancy). Qwen-14B fp32 on an 80 GB A100 holds 28k
   KV tokens, i.e. about 100 short GSM8K sequences, which gives r of about 0.3. Small r needs long
   outputs (MMLU-Pro, sec. 38), a smaller KV budget or a deliberately small batch, not just a larger
   model.

## 37. Perplexity check of the dequantized models (`scripts/ppl_check.sbatch`, `src/ppl_hf.py`)

WikiText-2 raw test (Salesforce/wikitext @ b08601e), n_ctx 512, llama-perplexity's chunking and
scoring (second half of each chunk) reproduced in `ppl_hf.py`. llama.cpp @ the pinned build, CUDA,
all layers on an A100. HF = dequantized fp32 safetensors under transformers (Track A weights).

| Version | Qwen-3B llama.cpp | Qwen-3B HF (Track A) | gap | Llama-8B llama.cpp | Llama-8B HF | gap |
|---|---|---|---|---|---|---|
| parent (bf16.gguf vs HF) | 8.9550 | 8.9549 | 0.00% | 7.1948 | 7.1947 | 0.00% |
| Q8_0 | 8.9767 | 8.9549 | +0.24% | 7.2026 | 7.1982 | +0.06% |
| Q6_K | 9.0423 | 9.0180 | +0.27% | 7.2291 | 7.2280 | +0.02% |
| Q4_K_M | 9.4857 | 9.4539 | +0.34% | 7.4033 | 7.3952 | +0.11% |
| Q3_K_M | 15.532 | 16.251 | -4.4% | 7.8623 | 7.8540 | +0.11% |

1. The pipeline is faithful: the parents agree to 1e-4 with the same token count (298,938 / 288,937)
   and chunk count, so tokenization, chunking and scoring match llama.cpp exactly.
2. Track A isolates weight quantization, and deployed GGUF inference adds more error. The most
   likely source is that llama.cpp's quantized CUDA matmul paths (MMQ / MMVQ) also quantize the
   activations to 8 bits (q8_1); the fp32 dequantized model does not. This is not verified from this
   build's logs. For Qwen-3B Q8_0, weight quantization alone moves perplexity by 0.0002% (8.95486 -> 8.95488),
   while the deployed GGUF moves it by 0.24%, a thousand times more. Llama-8B is much less sensitive
   (+0.05% vs +0.11%). So a Track A certificate is about the weight-quantized version, and Track B can
   show more divergence and more flips than Track A, especially for Qwen.
3. Perplexity barely registers Q8_0 even though the token-level outputs diverge on half of the Qwen-3B
   GSM8K questions (|A|/|P| 0.519). Version comparison by perplexity says little about
   finite-population answer flips.

## 38. MMLU-Pro pilot (1,000) for Llama-3.1-8B (job 2281326) and Qwen-14B (job 2281325), fp32, eager

Llama-3.1-8B: parent accuracy 49.3%, mean output 425 tokens. The local decoder matches vLLM's text on
999 / 1,000 outputs (used only by the text-class analyses).

| Version | Child acc. | \|A\|/\|P\| | harm / repair | sigma | law obs/pred | lemma 3 violations | t\* median |
|---|---|---|---|---|---|---|---|
| Q8_0 | 48.9% | **0.696** | 50 / 46 | 0.058 | 0.86 | 0 | 81 |
| Q6_K | 50.3% | 0.935 | 71 / 81 | 0.19 | 0.85 | 0 | 40 |
| Q4_K_M | 48.2% | 0.996 | 108 / 97 | 0.54 | 0.85 | 0 | 18 |
| Q3_K_M | 44.4% | 1.000 | 128 / 79 | 0.91 | 0.88 | 0 | 11 |

No flips outside D; one parent-side tie in the Q6_K run (resolved, no rule violations). f_m(0) is
about 0.18 (1.79% of margins < 0.1), the highest of all pairs, and the outputs are long, so screening
is weak. Divergences change the answer often (Q8_0: 68% keep it, vs 85% on GSM8K). Flips are close to
symmetric. Equal budget (harm, true worst-slice H 0.088 for Q8_0): at r = 0.076 screening is exact at
b = 0.4 (B1 0.099, B6 0.094 with coverage 0.80, B2 0.109). The gain is small because H is far above
tau, and every method reaches the truth at b = 0.6. Measured r in the truth run: 0.34.
Margin law, 36 cases now: all ratios 0.72-1.14.

## Not yet measured

Baselines B3-B5 (design-only variants, see sec. 35), a betting-based bound for B2, track B (deployed
GGUF kernels), imatrix ladders, variance-type lower bound for the net certificate.
