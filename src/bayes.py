"""
Author: Jake Vick
Purpose: Bayesian Student-t (PyMC) on the full sample of log returns, with VaR / CVaR
point estimates, credible intervals and MCMC diagnostics (R-hat, bulk / tail ESS).

Model (fit on x / s, where s = sample std of the returns, then mapped back):
    mu_z         ~ Normal(0, 0.5)          -> mu    ~ Normal(0, 0.5 * s)
    sigma_z      ~ HalfNormal(1)           -> sigma ~ HalfNormal(s)
    nu_minus_2   ~ Gamma(alpha=2, beta=0.1) -> nu = 2 + nu_minus_2  (nu > 2, prior mean 22)
    x_t / s      ~ StudentT(nu, mu_z, sigma_z)

Why these priors (weakly informative, scaled to the data):
- mu: daily drift is tiny next to daily volatility; Normal(0, half a std) allows any
  plausible drift without letting the mean wander into the tails.
- sigma: the t scale is below the std (std = sigma * sqrt(nu / (nu - 2))), so a
  HalfNormal with scale = 1 std puts most mass on sensible values, not tight.
- nu: shifting by 2 keeps the variance finite (same reason as DF_FLOOR = 2.01 in
  distributions.py) without a hard floor at the boundary. Gamma(2, 0.1) is the common
  weakly informative choice for t degrees of freedom (Juarez & Steel 2010): it covers
  very heavy tails (nu near 3) through near-Normal (nu > 30).
- Sampling on x / s only rescales (no centering), so the priors stay centered at 0 and
  NUTS sees parameters of order 1 (better step-size adaptation, no overflow warnings).

VaR / CVaR:
- Per draw: the closed-form Student-t VaR / CVaR (same formulas as risk_metrics.py) for
  each posterior draw (mu, sigma, nu). This gives a posterior distribution of VaR and
  CVaR, summarized by mean, median, sd and an HDI (parameter uncertainty).
- Posterior predictive: simulate future returns from every draw (n_ppc per draw) and take
  the empirical alpha-quantile and the mean below it. This single number integrates over
  parameter uncertainty and is the Bayesian "point" VaR / CVaR.

Educational project, not investment advice.
"""

from pathlib import Path
import argparse
import os
import time
import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import t

from .risk_metrics import compute_var_student_t, compute_cvar_student_t

RHAT_MAX = 1.01
"""Warn if any R-hat is above this (Vehtari et al. 2021)."""

ESS_MIN = 400
"""Warn if any bulk or tail ESS is below this (about 100 per chain for 4 chains)."""

PARAMS = ["mu", "sigma", "nu"]


def hdi(x: np.ndarray, prob: float = 0.94) -> tuple:
    """Highest density interval: the narrowest interval holding `prob` of the samples.

    Parameters ----------
    x: 1-d array of samples
    prob: Interval mass (default 0.94, the ArviZ default)

    Output ----------
    (low, high)
    """
    if not 0 < prob < 1:
        raise ValueError(f"prob must be in (0, 1) (got {prob})")
    s = np.sort(np.asarray(x, dtype=float).ravel())
    n = s.size
    k = int(np.floor(prob * n))
    if n == 0 or k < 1:
        raise ValueError("hdi: not enough samples")
    widths = s[k:] - s[: n - k]
    i = int(np.argmin(widths))
    return float(s[i]), float(s[i + k])


def build_model(returns, scale: Optional[float] = None):
    """Build the PyMC Student-t model on returns / scale (see module docstring).

    Parameters ----------
    returns: Log returns (Series or array); non-finite values are dropped
    scale: Rescaling constant s (default: sample std of returns)

    Output ----------
    (model, s)
    """
    import pymc as pm

    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        raise ValueError("bayes: need at least 3 finite returns")
    s = float(np.std(x)) if scale is None else float(scale)
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"bayes: returns have zero or invalid std ({s})")

    with pm.Model() as model:
        mu_z = pm.Normal("mu_z", mu=0.0, sigma=0.5)
        sigma_z = pm.HalfNormal("sigma_z", sigma=1.0)
        nu_minus_2 = pm.Gamma("nu_minus_2", alpha=2.0, beta=0.1)
        nu = pm.Deterministic("nu", nu_minus_2 + 2.0)
        pm.Deterministic("mu", mu_z * s)
        pm.Deterministic("sigma", sigma_z * s)
        pm.StudentT("obs", nu=nu, mu=mu_z, sigma=sigma_z, observed=x / s)
    return model, s


def fit_bayes_student_t(
    returns,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    cores: Optional[int] = None,
    seed: int = 42,
    progressbar: bool = False,
):
    """Sample the posterior with NUTS. Seeded, so the same seed and inputs give the same draws.

    Output ----------
    ArviZ DataTree / InferenceData returned by pm.sample
    """
    import pymc as pm

    if cores is None:
        cores = min(chains, os.cpu_count() or 1)
    model, _ = build_model(returns)
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            random_seed=seed,
            progressbar=progressbar,
            compute_convergence_checks=False,  # we report our own diagnostics
        )
    return idata


def posterior_draws(idata) -> Dict[str, np.ndarray]:
    """Flatten posterior draws of mu, sigma, nu to 1-d arrays (chain-major)."""
    post = idata.posterior
    return {p: np.asarray(post[p].values, dtype=float).ravel() for p in PARAMS}


def diagnostics(idata) -> pd.DataFrame:
    """R-hat and bulk / tail ESS for mu, sigma, nu (rank-normalized, ArviZ)."""
    import arviz as az

    post = idata.posterior
    ds = post.to_dataset() if hasattr(post, "to_dataset") else post
    ds = ds[PARAMS]
    rhat = az.rhat(ds)
    ess_bulk = az.ess(ds, method="bulk")
    ess_tail = az.ess(ds, method="tail")
    return pd.DataFrame(
        {
            "r_hat": [float(rhat[p]) for p in PARAMS],
            "ess_bulk": [float(ess_bulk[p]) for p in PARAMS],
            "ess_tail": [float(ess_tail[p]) for p in PARAMS],
        },
        index=PARAMS,
    )


def per_draw_var_cvar(mu: np.ndarray, sigma: np.ndarray, nu: np.ndarray, alpha: float):
    """Closed-form Student-t VaR and CVaR for each posterior draw (reuses risk_metrics)."""
    var = np.array([compute_var_student_t(n, m, s, alpha) for m, s, n in zip(mu, sigma, nu)])
    cvar = np.array([compute_cvar_student_t(n, m, s, alpha) for m, s, n in zip(mu, sigma, nu)])
    return var, cvar


def predictive_var_cvar(mu, sigma, nu, alpha: float, n_ppc: int = 50, seed: int = 42):
    """Posterior predictive VaR / CVaR: simulate n_ppc returns per draw, then take the
    empirical alpha-quantile and the mean of returns at or below it."""
    rng = np.random.default_rng(seed)
    z = t.rvs(np.repeat(nu, n_ppc), size=nu.size * n_ppc, random_state=rng)
    y = np.repeat(mu, n_ppc) + np.repeat(sigma, n_ppc) * z
    var = float(np.quantile(y, alpha))
    cvar = float(y[y <= var].mean())
    return var, cvar


def summarize(idata, alpha: float = 0.05, hdi_prob: float = 0.94, n_ppc: int = 50,
              seed: int = 42, n_obs: Optional[int] = None) -> pd.DataFrame:
    """Posterior summary table: one row per quantity (mu, sigma, nu, var, cvar).

    Columns: mean, sd, median, hdi_low, hdi_high, predictive (var/cvar only), r_hat,
    ess_bulk, ess_tail (params only), plus run metadata. Layout is in docs/SCHEMA.md.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1) (got {alpha})")
    d = posterior_draws(idata)
    var, cvar = per_draw_var_cvar(d["mu"], d["sigma"], d["nu"], alpha)
    pred_var, pred_cvar = predictive_var_cvar(d["mu"], d["sigma"], d["nu"], alpha, n_ppc, seed)
    diag = diagnostics(idata)

    samples = {**d, "var": var, "cvar": cvar}
    rows = []
    for q, x in samples.items():
        lo, hi = hdi(x, hdi_prob)
        rows.append({
            "quantity": q,
            "mean": float(np.mean(x)),
            "sd": float(np.std(x, ddof=1)),
            "median": float(np.median(x)),
            "hdi_low": lo,
            "hdi_high": hi,
            "predictive": {"var": pred_var, "cvar": pred_cvar}.get(q, np.nan),
            "r_hat": diag.loc[q, "r_hat"] if q in diag.index else np.nan,
            "ess_bulk": diag.loc[q, "ess_bulk"] if q in diag.index else np.nan,
            "ess_tail": diag.loc[q, "ess_tail"] if q in diag.index else np.nan,
        })
    out = pd.DataFrame(rows).set_index("quantity")

    post = idata.posterior
    out["alpha"] = alpha
    out["hdi_prob"] = hdi_prob
    out["n_obs"] = n_obs if n_obs is not None else np.nan
    out["chains"] = int(post.sizes["chain"])
    out["draws"] = int(post.sizes["draw"])
    out["seed"] = seed
    out["divergences"] = int(np.asarray(idata.sample_stats["diverging"].values).sum())
    return out


def check_diagnostics(summary: pd.DataFrame, rhat_max: float = RHAT_MAX, ess_min: float = ESS_MIN) -> list:
    """Return a list of diagnostic problems (empty if fine) and warn for each one."""
    problems = []
    p = summary.loc[PARAMS]
    for q, row in p.iterrows():
        if not np.isfinite(row["r_hat"]) or row["r_hat"] > rhat_max:
            problems.append(f"R-hat for {q} is {row['r_hat']:.4f} (> {rhat_max})")
        for col in ("ess_bulk", "ess_tail"):
            if not np.isfinite(row[col]) or row[col] < ess_min:
                problems.append(f"{col} for {q} is {row[col]:.0f} (< {ess_min})")
    div = int(summary["divergences"].iloc[0])
    if div > 0:
        problems.append(f"{div} divergent transitions")
    for msg in problems:
        warnings.warn(f"Bayes diagnostics: {msg}", RuntimeWarning)
    return problems


def run_bayes(returns, alpha: float = 0.05, draws: int = 1000, tune: int = 1000, chains: int = 4,
              seed: int = 42, hdi_prob: float = 0.94, n_ppc: int = 50, cores: Optional[int] = None,
              progressbar: bool = False):
    """Fit + summarize + check. Returns (summary_df, runtime_seconds)."""
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    t0 = time.perf_counter()
    idata = fit_bayes_student_t(x, draws=draws, tune=tune, chains=chains, cores=cores,
                                seed=seed, progressbar=progressbar)
    summary = summarize(idata, alpha=alpha, hdi_prob=hdi_prob, n_ppc=n_ppc, seed=seed, n_obs=x.size)
    runtime = time.perf_counter() - t0
    check_diagnostics(summary)
    return summary, runtime


def print_bayes_summary(summary: pd.DataFrame, runtime: Optional[float] = None) -> None:
    """Console print of the Bayesian results (VaR / CVaR, intervals, params, diagnostics)."""
    hp = int(round(summary["hdi_prob"].iloc[0] * 100))
    v, c = summary.loc["var"], summary.loc["cvar"]
    print(f"Bayes Student-t VaR (predictive):  {v['predictive']:.4f}   "
          f"posterior mean {v['mean']:.4f}, {hp}% HDI [{v['hdi_low']:.4f}, {v['hdi_high']:.4f}]")
    print(f"Bayes Student-t CVaR (predictive): {c['predictive']:.4f}   "
          f"posterior mean {c['mean']:.4f}, {hp}% HDI [{c['hdi_low']:.4f}, {c['hdi_high']:.4f}]")
    print(f"\nPosterior ({int(summary['chains'].iloc[0])} chains x {int(summary['draws'].iloc[0])} draws, "
          f"seed {int(summary['seed'].iloc[0])}, divergences {int(summary['divergences'].iloc[0])}):")
    cols = ["mean", "sd", "hdi_low", "hdi_high", "r_hat", "ess_bulk", "ess_tail"]
    print(summary.loc[PARAMS, cols].to_string(float_format=lambda f: f"{f:.4f}"))
    if runtime is not None:
        print(f"Bayes sampling + summary took {runtime:.1f} s")


if __name__ == "__main__":
    """
    Run with:
        python -m src.bayes --coin_symbol btc
    """
    parser = argparse.ArgumentParser(description="Bayesian Student-t VaR / CVaR on the existing returns CSV")
    parser.add_argument("--coin_symbol", type=str, default="btc", help="Coin symbol (e.g., btc, eth, sol)")
    parser.add_argument("--alpha", type=float, default=0.05, help="Tail probability (default: 0.05)")
    parser.add_argument("--draws", type=int, default=1000, help="Posterior draws per chain (default: 1000)")
    parser.add_argument("--tune", type=int, default=1000, help="Tuning steps per chain (default: 1000)")
    parser.add_argument("--chains", type=int, default=4, help="MCMC chains (default: 4)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    returns_path = base_dir / "data" / "output" / f"{args.coin_symbol}-log-returns.csv"
    returns = pd.read_csv(returns_path)["return"]
    summary, runtime = run_bayes(returns, alpha=args.alpha, draws=args.draws, tune=args.tune,
                                 chains=args.chains, seed=args.seed)
    print_bayes_summary(summary, runtime)
