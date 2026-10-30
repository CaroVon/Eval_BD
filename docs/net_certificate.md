# Net-degradation certificate (draft)

Status: draft for discussion, written 2026-10-30. Not part of plan v4.0; motivated by the
near-symmetric harm/repair flips in the pilot (RESULTS.md sec. 12).

## Setting

As in plan sec. 1.3: slice g with N_g questions, parent-correct set P, flips F_i in {0, 1},

    theta_g = H_g - Rep_g,   H_g = (1/N_g) sum_{i in g n P} F_i,   Rep_g = (1/N_g) sum_{i in g \ P} F_i.

The plan certifies theta_g through theta_g <= H_g (harm only). When repairs are as frequent as
harms, this bound is loose by Rep_g, which can exceed tau even when theta_g is near 0
(MMLU-Pro Q8_0: H = 5.3%, theta = 0.7%).

## Exact screening applies to both sides

Lemma 3 holds for every question, not only for i in P: if i is not in D_q, the child output equals
the parent output and F_i = 0. So harm candidates are A = g n P n D and repair candidates are
B = g \ P n D.

## Lower bound for the repair count

Let T = sum_{i in B} F_i, independent R_i ~ Bern(pi_i) with pi_i <= pi_max on B, and
K = sum_{i in B} R_i F_i.

**Lemma (stochastic upper bound).** P(K >= k) <= P(Bin(T, pi_max) >= k) for every k.
*Proof.* Mirror of lemma 2: with V_i uniform, R_i = 1[V_i <= pi_i] <= R'_i = 1[V_i <= pi_max],
so K <= K' ~ Bin(T, pi_max) pointwise.

**Proposition (exact lower bound).** Let L(t, k) = P(Bin(t, pi_max) >= k) and
T_L(k) = min{t in 0..|B| : L(t, k) > delta} (= |B| if no such t). Then for every configuration F,
P(T < T_L(K)) <= delta.
*Proof.* L is non-decreasing in t. If T < T_L(k) then L(T, k) <= delta. Let
k* = min{k : L(T, k) <= delta} (if none, the failure event is empty). Since L(T, k) is
non-increasing in k, {T < T_L(K)} is contained in {K >= k*}, and by the lemma
P(K >= k*) <= L(T, k*) <= delta.

## Net certificate

**Theorem.** With T_U from theorem 4 applied to A at level delta'/2 and T_L applied to B at level
delta'/2, UCB^net_g = (T_U(K_A) - T_L(K_B)) / N_g satisfies P(theta_g <= UCB^net_g) >= 1 - delta'.
Uniformity over versions x slices follows from the union bound as in theorem 5.
*Proof.* theta_g = (T_A - T_B)/N_g, and the two failure events each have probability <= delta'/2.

Implementation: `src/cert.py` (`tl_table`), `src/simulate_net.py`. A Monte Carlo check of
`tl_table` gives failure rates 0.047, 0.009, 0.031, 0 at nominal 0.05, 0.01, 0.05, 0.05.

## What it costs

The net certificate also needs D on parent-incorrect questions (screen all questions) and audits
of B. Simulations (RESULTS.md sec. 15) show it is the only way to certify Q6_K on GSM8K and Q8_0
or Q6_K on MMLU-Pro at tau = 0.05, but only with inclusion probabilities of 0.9-0.99, i.e. close to
the cost of full evaluation.

## Open questions

- Matching lower bound for auditing cost of net certification (analogue of theorems 7-8): the
  two-point argument now needs configurations that move harm and repair counts together.
- Joint design of pi on A and B (currently the zero-harm cover design with a floor).
- Whether a correlated (paired) design across harm and repair candidates tightens the bound.
