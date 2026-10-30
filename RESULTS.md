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

## Not yet measured

Chunk verification cost, MMLU-Pro 3,000, second parent family, remaining baselines (B2-B6),
track B, perplexity check of the dequantized models.
