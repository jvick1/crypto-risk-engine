"""
Author: Jake Vick
Purpose: Rolling Bayesian VaR / CVaR for the out-of-sample backtest (Phase 4).

Two engines, picked with --bayes_engine:

mcmc (default): the Phase 3 Bayesian Student-t (same priors as src/bayes.py), sampled with
    PyMC NUTS on each refit window. The model is compiled once and the window is swapped in
    with pm.set_data, so a refit costs about 1 s on the box (2 chains x 500 draws + 500 tune).
    VaR / CVaR are the exact posterior predictive values (mixture of per-draw t's), the same
    as Phase 3's "predictive" column.

conjugate: a Normal likelihood with a Normal-Inverse-Gamma (NIG) prior. The posterior is
    closed form and the posterior predictive of tomorrow's return is a Student-t, so VaR /
    CVaR are closed form and the engine is fast enough to refit every day. Because the
    likelihood is Normal, its fat tails come only from parameter uncertainty; with a 252-day
    window the predictive df is about 2 * (a0 + 126) = 254, i.e. essentially Normal.

Refit schedule (no look-ahead):
    Forecast days are i = window, window + 1, ..., n - 1 (0-based positions in the sorted
    returns). Refits happen on days r = window, window + k, window + 2k, ... (k =
    --refit_every). The fit for refit day r uses only returns[r - window : r], i.e. data that
    ends the day before r. Every forecast day i in [r, r + k) reuses that posterior. So a
    forecast for day i never uses returns[i:], and with k > 1 the posterior can be up to
    k - 1 days stale. With k = 1 the window is exactly returns[i - window : i], like the
    other backtest methods. The schedule is anchored at the first forecast day, so appending
    new data keeps every earlier refit (and its cache entry) unchanged.

Warm start (mcmc): each refit starts NUTS at the previous refit's posterior means (mapped to
    the new window's scale), rounded to 6 significant digits. The priors are NOT replaced by
    the previous posterior: consecutive windows overlap by window - k days, so using the old
    posterior as the new prior would count those days twice. Warm start only speeds up the
    sampler's start; each window's posterior depends on its own data and the fixed priors.

Cache (mcmc): each refit's posterior draws and diagnostics are saved under data/cache/
    (gitignored) as .npz files keyed by a SHA-256 of: the exact float64 bytes of the window,
    the model / prior version tag, draws, tune, chains, the per-refit seed and the warm-start
    init values. Any change to the data, the priors, the sampler settings, the seed or the
    init chain gives a new key, so a stale fit is never reused. alpha is not in the key
    because VaR / CVaR are recomputed from the cached draws. The conjugate engine is not
    cached: all ~4,000 daily refits take well under a second, faster than reading files.

Educational project, not investment advice.
"""

from pathlib import Path
import hashlib
import logging
import os
import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .risk_metrics import compute_var_student_t, compute_cvar_student_t

ENGINES = ("mcmc", "conjugate")
DEFAULT_REFIT_EVERY = {"mcmc": 21, "conjugate": 1}
"""mcmc: about one month of trading days; a 252-day window changes by only 21/252 (8%) of
its data between refits, and ~190 refits keep a full BTC backtest to a few minutes.
conjugate: closed form, so refit every day."""

# Rolling MCMC sampler defaults (smaller than the full-sample run: 1 window = 252 points)
BT_DRAWS = 500
BT_TUNE = 500
BT_CHAINS = 2

RHAT_MAX = 1.01
ESS_MIN_PER_CHAIN = 100
"""Rolling refits fail the check if any R-hat > 1.01, any bulk/tail ESS < 100 per chain
(Vehtari et al. 2021), or any divergent transition."""

# Normal-Inverse-Gamma prior for the conjugate engine (on raw daily log returns):
#   sigma^2 ~ InvGamma(a0, b0),  mu | sigma^2 ~ Normal(m0, sigma^2 / kappa0)
NIG_M0 = 0.0      # prior mean drift 0
NIG_KAPPA0 = 1.0  # worth 1 pseudo-observation for the mean
NIG_A0 = 1.0      # 2 * a0 = 2 pseudo-observations for the variance ...
NIG_B0 = 1e-4     # ... with b0 / a0 = 1e-4, i.e. a 1% daily volatility guess

MODEL_TAG = "bayes-student-t-v1|mu_z~N(0,0.5)|sigma_z~HN(1)|nu=2+Gamma(2,0.1)|x/std(ddof=0)"
"""Changes to the priors / parameterization must change this tag (invalidates the cache)."""

CACHE_VERSION = "1"


# ---------------------------------------------------------------- schedule

def refit_schedule(n: int, window_size: int, refit_every: int) -> np.ndarray:
    """Refit day (0-based position) whose posterior is used for each forecast day.

    Output: int array of length n - window_size; entry j is the refit day for forecast
    day i = window_size + j, i.e. window_size + floor(j / k) * k (always <= i).
    """
    if refit_every < 1:
        raise ValueError(f"refit_every must be >= 1 (got {refit_every})")
    if n <= window_size:
        raise ValueError(f"need more than window_size={window_size} returns (got {n})")
    j = np.arange(n - window_size)
    return window_size + (j // refit_every) * refit_every


# ---------------------------------------------------------------- conjugate engine

def nig_posterior(x, m0: float = NIG_M0, kappa0: float = NIG_KAPPA0,
                  a0: float = NIG_A0, b0: float = NIG_B0) -> tuple:
    """Normal-Inverse-Gamma posterior for a Normal likelihood with unknown mean and variance.

        kappa_n = kappa0 + n
        m_n     = (kappa0 * m0 + n * xbar) / kappa_n
        a_n     = a0 + n / 2
        b_n     = b0 + 0.5 * sum((x - xbar)^2) + kappa0 * n * (xbar - m0)^2 / (2 * kappa_n)

    Output: (m_n, kappa_n, a_n, b_n)
    """
    x = np.asarray(x, dtype=float)
    if x.size < 1 or not np.isfinite(x).all():
        raise ValueError("nig_posterior: need at least one finite return and no NaN/inf")
    if kappa0 <= 0 or a0 <= 0 or b0 <= 0:
        raise ValueError("nig_posterior: kappa0, a0 and b0 must be > 0")
    n = x.size
    xbar = float(x.mean())
    ss = float(np.sum((x - xbar) ** 2))
    kappa_n = kappa0 + n
    m_n = (kappa0 * m0 + n * xbar) / kappa_n
    a_n = a0 + n / 2.0
    b_n = b0 + 0.5 * ss + kappa0 * n * (xbar - m0) ** 2 / (2.0 * kappa_n)
    return m_n, kappa_n, a_n, b_n


def nig_predictive(m_n: float, kappa_n: float, a_n: float, b_n: float) -> tuple:
    """Posterior predictive of one new return: Student-t(df = 2 a_n, loc = m_n,
    scale = sqrt(b_n (kappa_n + 1) / (a_n kappa_n))). Output: (df, loc, scale)."""
    return 2.0 * a_n, m_n, float(np.sqrt(b_n * (kappa_n + 1.0) / (a_n * kappa_n)))


def conjugate_var_cvar(x, alpha: float, **prior) -> tuple:
    """Closed-form predictive VaR / CVaR for one window (conjugate engine)."""
    df, loc, scale = nig_predictive(*nig_posterior(x, **prior))
    return compute_var_student_t(df, loc, scale, alpha), compute_cvar_student_t(df, loc, scale, alpha)


# ---------------------------------------------------------------- MCMC engine

def build_rolling_model(window_size: int):
    """Phase 3 Student-t model with the window in pm.Data, so it compiles once.

    Same priors as src.bayes.build_model; s (the window std, ddof=0) is data too.
    """
    import pymc as pm

    with pm.Model() as model:
        xz = pm.Data("xz", np.zeros(window_size))
        s = pm.Data("s", 1.0)
        mu_z = pm.Normal("mu_z", mu=0.0, sigma=0.5)
        sigma_z = pm.HalfNormal("sigma_z", sigma=1.0)
        nu_minus_2 = pm.Gamma("nu_minus_2", alpha=2.0, beta=0.1)
        nu = pm.Deterministic("nu", nu_minus_2 + 2.0)
        pm.Deterministic("mu", mu_z * s)
        pm.Deterministic("sigma", sigma_z * s)
        pm.StudentT("obs", nu=nu, mu=mu_z, sigma=sigma_z, observed=xz)
    return model


def _round_sig(v: float, sig: int = 6) -> float:
    return float(f"{v:.{sig}g}")


def warm_init(prev_means: Optional[dict], s: float) -> Optional[dict]:
    """Map previous posterior means (raw scale) to init values on the new window's scale."""
    if prev_means is None:
        return None
    return {
        "mu_z": _round_sig(prev_means["mu"] / s),
        "sigma_z": _round_sig(max(prev_means["sigma"] / s, 1e-3)),
        "nu_minus_2": _round_sig(max(prev_means["nu"] - 2.0, 1e-3)),
    }


def cache_key(window: np.ndarray, draws: int, tune: int, chains: int, seed: int,
              init: Optional[dict]) -> str:
    """SHA-256 over everything that changes an MCMC refit's draws."""
    h = hashlib.sha256()
    h.update(f"v{CACHE_VERSION}|{MODEL_TAG}|d{draws}|t{tune}|c{chains}|s{seed}|".encode())
    h.update(repr(sorted(init.items()) if init else None).encode())
    h.update(np.ascontiguousarray(window, dtype=np.float64).tobytes())
    return h.hexdigest()


def load_cached(cache_dir: Optional[Path], key: str) -> Optional[dict]:
    if cache_dir is None:
        return None
    path = Path(cache_dir) / f"{key}.npz"
    if not path.exists():
        return None
    try:
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    except Exception:  # corrupt / partial file: treat as a miss and refit
        return None


def save_cached(cache_dir: Optional[Path], key: str, fit: dict) -> None:
    if cache_dir is None:
        return
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_dir / f"{key}.tmp.npz"
    np.savez(tmp, **fit)
    os.replace(tmp, cache_dir / f"{key}.npz")  # atomic: no half-written cache entries


def sample_window(model, window: np.ndarray, draws: int, tune: int, chains: int, seed: int,
                  init: Optional[dict]) -> dict:
    """One NUTS refit on a window. Output: dict of draws (mu, sigma, nu) + diagnostics."""
    import pymc as pm
    from .bayes import PARAMS, diagnostics

    s = float(np.std(window))
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"window has zero or invalid std ({s})")
    with model:
        pm.set_data({"xz": window / s, "s": s})
        kwargs = {"initvals": init} if init else {}
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=1, random_seed=seed,
                          progressbar=False, compute_convergence_checks=False, **kwargs)
    post = idata.posterior
    diag = diagnostics(idata)
    fit = {p: np.asarray(post[p].values, dtype=float).ravel() for p in PARAMS}
    fit["r_hat"] = diag["r_hat"].to_numpy(dtype=float)
    fit["ess_bulk"] = diag["ess_bulk"].to_numpy(dtype=float)
    fit["ess_tail"] = diag["ess_tail"].to_numpy(dtype=float)
    fit["divergences"] = np.array(int(np.asarray(idata.sample_stats["diverging"].values).sum()))
    return fit


def diagnostic_problems(fit: dict, chains: int) -> list:
    """List of failed checks for one refit (empty = pass). No warnings are raised here."""
    problems = []
    rhat = np.asarray(fit["r_hat"], dtype=float)
    ess = np.minimum(np.asarray(fit["ess_bulk"], dtype=float), np.asarray(fit["ess_tail"], dtype=float))
    if not np.isfinite(rhat).all() or rhat.max() > RHAT_MAX:
        problems.append("rhat")
    if not np.isfinite(ess).all() or ess.min() < ESS_MIN_PER_CHAIN * chains:
        problems.append("ess")
    if int(fit["divergences"]) > 0:
        problems.append("divergences")
    return problems


# ---------------------------------------------------------------- driver

def rolling_bayes(returns, window_size: int = 252, alpha: float = 0.05, engine: str = "mcmc",
                  refit_every: Optional[int] = None, draws: int = BT_DRAWS, tune: int = BT_TUNE,
                  chains: int = BT_CHAINS, seed: int = 42, cache_dir: Optional[Path] = None,
                  progressbar: bool = True):
    """Rolling Bayesian VaR / CVaR for every forecast day.

    Parameters ----------
    returns: log returns sorted by date ascending (array-like, finite)
    engine: "mcmc" (Student-t, NUTS) or "conjugate" (Normal-Inverse-Gamma, closed form)
    refit_every: k, days between refits (default 21 for mcmc, 1 for conjugate)
    draws / tune / chains: NUTS settings per refit (mcmc only)
    seed: base seed; refit on day r uses seed + r
    cache_dir: directory for cached MCMC fits (None disables the cache)

    Output ----------
    (var, cvar, refits): two arrays of length n - window_size, and a DataFrame with one row
    per refit (schema in docs/SCHEMA.md)
    """
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES} (got {engine!r})")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1) (got {alpha})")
    k = DEFAULT_REFIT_EVERY[engine] if refit_every is None else int(refit_every)
    x = np.asarray(returns, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("rolling_bayes: returns contain NaN/inf")
    n = x.size
    sched = refit_schedule(n, window_size, k)
    refit_days = np.unique(sched)

    model = None
    pymc_logger = logging.getLogger("pymc")
    old_level = None
    if engine == "mcmc":
        from .bayes import predictive_var_cvar

    var_at, cvar_at, rows = {}, {}, []
    prev_means = None
    it = tqdm(refit_days, desc=f"Rolling Bayes ({engine}, k={k})", disable=not progressbar)
    for r in it:
        window = x[r - window_size:r]  # data ends the day before the refit day: no look-ahead
        t0 = time.perf_counter()
        row = {"refit_pos": int(r), "fit_start_pos": int(r - window_size), "fit_end_pos": int(r - 1),
               "n_obs": int(window.size)}
        if engine == "conjugate":
            m_n, kappa_n, a_n, b_n = nig_posterior(window)
            df, loc, scale = nig_predictive(m_n, kappa_n, a_n, b_n)
            v = compute_var_student_t(df, loc, scale, alpha)
            c = compute_cvar_student_t(df, loc, scale, alpha)
            row.update(mu_mean=m_n, sigma_mean=float(np.sqrt(b_n / (a_n - 1.0))) if a_n > 1 else np.nan,
                       nu_mean=np.nan, pred_df=df, pred_loc=loc, pred_scale=scale)
        else:
            s = float(np.std(window))
            init = warm_init(prev_means, s)
            seed_r = int(seed) + int(r)
            key = cache_key(window, draws, tune, chains, seed_r, init)
            fit = load_cached(cache_dir, key)
            cached = fit is not None
            if not cached:
                if model is None:  # compiled lazily, so an all-cached rerun never imports PyMC
                    model = build_rolling_model(window_size)  # imports pymc, which sets its log level
                    old_level = pymc_logger.level
                    pymc_logger.setLevel(logging.ERROR)  # no per-refit "Sequential sampling ..." lines
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # counted below instead of printed per refit
                    fit = sample_window(model, window, draws, tune, chains, seed_r, init)
                save_cached(cache_dir, key, fit)
            v, c = predictive_var_cvar(fit["mu"], fit["sigma"], fit["nu"], alpha)
            means = {p: float(np.mean(fit[p])) for p in ("mu", "sigma", "nu")}
            prev_means = means
            problems = diagnostic_problems(fit, chains)
            row.update(mu_mean=means["mu"], sigma_mean=means["sigma"], nu_mean=means["nu"],
                       r_hat_max=float(np.max(fit["r_hat"])),
                       ess_bulk_min=float(np.min(fit["ess_bulk"])),
                       ess_tail_min=float(np.min(fit["ess_tail"])),
                       divergences=int(fit["divergences"]),
                       passed=not problems, problems=",".join(problems),
                       cached=cached, seed=seed_r, cache_key=key[:16])
        row.update(var=v, cvar=c, seconds=time.perf_counter() - t0)
        var_at[r], cvar_at[r] = v, c
        rows.append(row)

    if old_level is not None:
        pymc_logger.setLevel(old_level)

    var = np.array([var_at[r] for r in sched])
    cvar = np.array([cvar_at[r] for r in sched])
    refits = pd.DataFrame(rows)
    refits.insert(0, "engine", engine)
    refits.insert(1, "refit_every", k)
    return var, cvar, refits


def refit_report(refits: pd.DataFrame) -> str:
    """One-paragraph console summary: refit count, cache hits, diagnostic failures."""
    engine = refits["engine"].iloc[0]
    k = int(refits["refit_every"].iloc[0])
    total = float(refits["seconds"].sum())
    msg = f"Rolling Bayes ({engine}, refit every {k} days): {len(refits)} refits in {total:.1f} s"
    if engine == "mcmc":
        failed = int((~refits["passed"].astype(bool)).sum())
        counts = {p: int(refits["problems"].str.contains(p).sum()) for p in ("rhat", "ess", "divergences")}
        msg += (f"; cache hits {int(refits['cached'].sum())}/{len(refits)}"
                f"; refits failing diagnostics {failed}/{len(refits)} "
                f"(R-hat > {RHAT_MAX}: {counts['rhat']}, ESS < {ESS_MIN_PER_CHAIN}/chain: {counts['ess']}, "
                f"with divergences: {counts['divergences']})")
    return msg
