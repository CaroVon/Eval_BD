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

## Two simple consequences

**Proposition (harm certificates cannot be tau-precise for theta when repairs are common).** Let
UCB be any upper bound that is valid for H_g (P(H_g <= UCB) >= 1 - delta for all configurations).
If Rep_g >= tau at configuration F^0, then P_{F^0}(UCB < theta_g + tau) <= delta.
*Proof.* theta_g + tau <= theta_g + Rep_g = H_g, so {UCB < theta_g + tau} is contained in
{UCB < H_g}, which has probability at most delta. This holds at every audit rate; it is the
situation of MMLU-Pro Q8_0 (Rep = 4.6%, so tau below about 0.046 is out of reach for a harm
certificate regardless of cost).

**Proposition (audit lower bound for the net certificate, Poisson designs).** Fix g, candidate sets
A (harm side) and B (repair side), inclusion probabilities pi_i fixed in advance, and a procedure
valid for theta_g that is tau-precise at F^0. Then
(i) for every U subset of A with F^0_U = 0 and |U| >= tau N_g: prod_{i in U} (1 - pi_i) <= 2 delta;
(ii) for every V subset of B with F^0_V = 1 and |V| >= tau N_g: prod_{i in V} (1 - pi_i) <= 2 delta.
*Proof.* (i) is theorem 7 (adding harms on U raises theta by |U|/N_g >= tau). For (ii), let
F^1 = F^0 - 1_V; removing repairs on V raises theta by |V|/N_g >= tau. On the event that no item
of V is audited, the observations under F^0 and F^1 coincide, and the rest of the argument of
theorem 7 applies unchanged.

With k = ceil(tau N_g), (ii) says every repair candidate that is actually a repair at F^0, except at
most k - 1, needs pi_i >= 1 - (2 delta)^{1/k}. Because the design is fixed before F is known, being
precise at every configuration with at least k repairs on B forces this rate on all of B (up to
k - 1 items), so the necessary expected audits grow from about |A| (1 - (2 delta)^{1/k}) to about
(|A| + |B|)(1 - (2 delta)^{1/k}). The extra price of certifying theta instead of H is the repair
candidate set B; exact screening shrinks it to g \ P n D, as it shrinks A.

These two-point bounds are of the same order as the zero-harm bound. They do not explain the
much higher rates (0.9-0.99) the simulations need when both sides carry many flips; that regime
needs a variance-type lower bound (open).

## Open questions

- Variance-type lower bound when harm and repair counts are both large (the regime of the
  simulations); the two-point bounds above only give the zero-flip rate.
- Joint design of pi on A and B (currently the zero-harm cover design with a floor).
- Whether a correlated (paired) design across harm and repair candidates tightens the bound.
