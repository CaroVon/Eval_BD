"""Exact one-sided binomial-inversion certificate and slice-cover design (plan sec 2.4).

Theorem 4: with inclusion probabilities >= pi on candidate set A (|A| = n) and observed
K = sum_{i in A} R_i F_i, the bound T <= T_U(K) holds with prob >= 1 - delta for every
configuration F, where T_U(k) = max{t in 0..n : P(Bin(t, pi) <= k) > delta}.
"""
import math

import numpy as np
from scipy.stats import binom


def tu_table(n, pi, delta):
    """T_U(k) for k = 0..n (theorem 4). pi = 0 gives the trivial bound n."""
    if n == 0:
        return np.zeros(1, dtype=int)
    if pi >= 1.0:
        return np.arange(n + 1)  # K = T exactly
    t = np.arange(n + 1)[:, None]
    k = np.arange(n + 1)[None, :]
    ok = binom.cdf(k, t, pi) > delta  # non-increasing in t, so a prefix in t for each k
    return ok.sum(axis=0) - 1


def tl_table(n, pi_max, delta):
    """Exact lower bound T_L(k) = min{t in 0..n : P(Bin(t, pi_max) >= k) > delta}, k = 0..n.

    Mirror of theorem 4 for a count T observed through K = sum R_i F_i with inclusion
    probabilities <= pi_max: K is stochastically dominated by Bin(T, pi_max), so
    P(T < T_L(K)) <= delta for every configuration.
    """
    if n == 0:
        return np.zeros(1, dtype=int)
    if pi_max >= 1.0:
        return np.arange(n + 1)
    t = np.arange(n + 1)[:, None]
    k = np.arange(n + 1)[None, :]
    ok = binom.sf(k - 1, t, pi_max) > delta  # P(Bin(t, pi) >= k), non-decreasing in t
    first = np.where(ok.any(axis=0), ok.argmax(axis=0), n)
    return first


def pi_star(n_g, tau, delta_prime):
    """Proposition 10: inclusion probability that makes slice g exactly tau-precise at zero harm."""
    k = math.ceil(tau * n_g)
    # At pi = 1 - delta'^(1/k) exactly, (1 - pi)^k = delta' and rounding can make it exceed
    # delta', giving T_U(0) = k. A tiny upward margin keeps T_U(0) <= k - 1; raising
    # inclusion probabilities never affects validity.
    return min(1.0, 1.0 - delta_prime ** (1.0 / k) + 1e-9)


def cover_design(slices, candidates, tau, delta_prime):
    """pi_i = max over slices containing i of pi*(g) for candidates i, else 0.

    slices: dict name -> bool array over questions; candidates: bool array.
    """
    n = len(candidates)
    pi = np.zeros(n)
    for mask in slices.values():
        pi[mask] = np.maximum(pi[mask], pi_star(int(mask.sum()), tau, delta_prime))
    pi[~candidates] = 0.0
    return np.minimum(pi, 1.0)


def zero_harm_ucb(n_cand, n_g, pi, delta_prime):
    """Deterministic certificate value when no candidate flips (K = 0 surely)."""
    return tu_table(n_cand, pi, delta_prime)[0] / n_g


def audit_rate_bounds(n_g, tau, delta, n_slices=1):
    """Table E1 of the plan: lower bound 1-(2 delta)^(1/k), single-slice and uniform sufficient rates."""
    k = math.ceil(tau * n_g)
    return {"k": k, "lower": 1 - (2 * delta) ** (1 / k), "single": 1 - delta ** (1 / k),
            "uniform": 1 - (delta / n_slices) ** (1 / k)}
