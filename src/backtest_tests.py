"""
Purpose: Statistical VaR backtest tests on a breach indicator series.

Implements:
- Kupiec proportion-of-failures (POF) unconditional coverage test, LR ~ chi2(1)
- Christoffersen independence test, LR ~ chi2(1)
- Christoffersen conditional coverage test, LR_cc = LR_pof + LR_ind ~ chi2(2)

Caveat: in a rolling backtest the estimation windows overlap, so consecutive
VaR forecasts are not independent. These tests assume the breach flags are
i.i.d. Bernoulli(alpha) under the null; treat p-values as approximate.
"""

from typing import Dict, Sequence
import numpy as np
from scipy.stats import chi2


def _xlogy(x: float, y: float) -> float:
    """x * log(y) with the convention 0 * log(0) = 0 (avoids NaN on edge cases)."""
    return 0.0 if x == 0 else x * np.log(y)


def _bernoulli_ll(n0: float, n1: float, p: float) -> float:
    """Log-likelihood of n0 non-breaches and n1 breaches with breach prob p."""
    return _xlogy(n0, 1.0 - p) + _xlogy(n1, p)


def kupiec_pof(breaches: Sequence[bool], alpha: float) -> Dict[str, float]:
    """
    Kupiec POF (unconditional coverage) test.

    H0: breach probability equals alpha. LR_pof ~ chi2(1).
    Zero breaches or all breaches are handled via 0*log(0) = 0.
    Assumes independent breach flags (overlapping rolling windows violate this).
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    b = np.asarray(breaches, dtype=bool)
    T = int(b.size)
    x = int(b.sum())
    if T == 0:
        return {"T": 0, "breaches": 0, "lr_pof": np.nan, "p_pof": np.nan}
    pi_hat = x / T
    lr = -2.0 * (_bernoulli_ll(T - x, x, alpha) - _bernoulli_ll(T - x, x, pi_hat))
    lr = max(lr, 0.0)
    return {"T": T, "breaches": x, "lr_pof": lr, "p_pof": float(chi2.sf(lr, 1))}


def christoffersen_independence(breaches: Sequence[bool]) -> Dict[str, float]:
    """
    Christoffersen independence test (first-order Markov vs i.i.d.).

    Counts transitions n_ij (state i at t-1 -> state j at t). LR_ind ~ chi2(1).
    If a state has no outgoing transitions (e.g. zero breaches, or the only
    breach is the last observation), its term contributes 0 and LR_ind = 0.
    Overlapping rolling windows induce dependence; p-values are approximate.
    """
    b = np.asarray(breaches, dtype=int)
    if b.size < 2:
        return {"n00": 0, "n01": 0, "n10": 0, "n11": 0, "lr_ind": 0.0, "p_ind": 1.0}
    prev, curr = b[:-1], b[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))
    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    n = n00 + n01 + n10 + n11
    pi = (n01 + n11) / n
    ll_null = _bernoulli_ll(n00 + n10, n01 + n11, pi)
    ll_alt = _bernoulli_ll(n00, n01, pi0) + _bernoulli_ll(n10, n11, pi1)
    lr = max(-2.0 * (ll_null - ll_alt), 0.0)
    return {"n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "lr_ind": lr, "p_ind": float(chi2.sf(lr, 1))}


def christoffersen_cc(breaches: Sequence[bool], alpha: float, sig: float = 0.05) -> Dict[str, float]:
    """
    Full summary: Kupiec POF, Christoffersen independence, and conditional
    coverage LR_cc = LR_pof + LR_ind ~ chi2(2), with reject flags at `sig`.
    Same independence caveat as above (overlapping rolling windows).
    """
    pof = kupiec_pof(breaches, alpha)
    ind = christoffersen_independence(breaches)
    lr_cc = pof["lr_pof"] + ind["lr_ind"]
    p_cc = float(chi2.sf(lr_cc, 2)) if np.isfinite(lr_cc) else np.nan
    T = pof["T"]
    return {
        "breaches": pof["breaches"],
        "total_periods": T,
        "expected_breaches": alpha * T,
        "breach_rate": pof["breaches"] / T if T else np.nan,
        "expected_rate": alpha,
        "lr_pof": pof["lr_pof"], "p_pof": pof["p_pof"],
        "lr_ind": ind["lr_ind"], "p_ind": ind["p_ind"],
        "lr_cc": lr_cc, "p_cc": p_cc,
        "reject_pof_5pct": bool(pof["p_pof"] < sig),
        "reject_ind_5pct": bool(ind["p_ind"] < sig),
        "reject_cc_5pct": bool(p_cc < sig),
    }
