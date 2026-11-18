# Technical Requirements: Crypto Risk Engine

> Educational project, not investment advice. Describes the code on `feat/phase3-bayesian-student-t`. Planned items are marked **Planned**.

## Architecture (by module)
A flat Python package `src/`, run as `python -m src.main`. Each module also has a standalone `__main__` for debugging.

| Module | Responsibility | Key functions |
|---|---|---|
| `src/api.py` | Extract: download daily data from Yahoo Finance via `yfinance` (`period="max"`, `interval="1d"`, `auto_adjust=False`), keep `Close`, save raw CSV | `fetch_historical_data(ticker)`, `save_raw_data(df, path)` |
| `src/returns.py` | Transform: parse UTC timestamps, sort, drop duplicate dates, compute `log(p_t / p_{t-1})`, drop first NaN, save `date,return` | `compute_log_returns(...)`, `print_return_summary(r)` |
| `src/distributions.py` | Fit Normal (`mean`, `np.std` i.e. ddof=0) and Student-t (`scipy.stats.t.fit`) with df floor `DF_FLOOR = 2.01` (refit with `f0=2.01` if the MLE df is lower) and Normal-equivalent fallback (`DF_NORMAL_FALLBACK = 1e6`) plus a `RuntimeWarning` if the fit fails | `fit_normal`, `fit_student_t` |
| `src/risk_metrics.py` | VaR / CVaR (left-tail quantiles of log returns, negative = loss) for Normal, Student-t (closed-form ES; raises `ValueError` for df <= 1, bad scale or alpha) and Historical (`np.percentile`, mean of returns <= VaR) | `compute_var_*`, `compute_cvar_*` |
| `src/backtest.py` | Rolling window backtest: for each day `i >= window_size`, fit on returns `[i-window_size, i)` and forecast day `i`; breach = `actual_return < VaR`. Normal uses pandas rolling mean/std (ddof=1); Student-t refits each window; Historical uses the window's empirical quantile | `rolling_backtest(df_returns, window_size, alpha, methods)` -> `(results_df, summary_df)` |
| `src/backtest_tests.py` | Kupiec POF (chi2, 1 df), Christoffersen independence (chi2, 1 df), conditional coverage `LR_cc = LR_pof + LR_ind` (chi2, 2 df), reject flags at 5%; `0*log(0)=0` for edge cases | `kupiec_pof`, `christoffersen_independence`, `christoffersen_cc` |
| `src/bayes.py` | Bayesian Student-t on the full sample (PyMC NUTS). Samples on `x / s` (s = sample std, no centering) with priors `mu_z ~ Normal(0, 0.5)`, `sigma_z ~ HalfNormal(1)`, `nu = 2 + Gamma(2, 0.1)`, i.e. `mu ~ Normal(0, 0.5 s)`, `sigma ~ HalfNormal(s)` on the raw scale. Per posterior draw: closed-form t VaR / CVaR (reuses `risk_metrics`), summarized by mean, sd, median and HDI (own narrowest-interval `hdi`, default 94%). Posterior predictive VaR / CVaR computed exactly from the mixture of per-draw t's (`brentq` on the mixture CDF + closed-form t partial expectation; no simulation noise). Diagnostics: ArviZ rank-normalized R-hat and bulk / tail ESS for mu, sigma, nu, plus divergence count; `RuntimeWarning` if R-hat > `RHAT_MAX` (1.01), ESS < `ESS_MIN` (400) or any divergence | `build_model`, `fit_bayes_student_t`, `summarize`, `predictive_var_cvar`, `diagnostics`, `check_diagnostics`, `run_bayes`, `print_bayes_summary` |
| `src/visualization.py` | Matplotlib/seaborn figures: histogram + fitted PDFs with VaR lines, QQ plots, log-scale left-tail density, rolling VaR with breach markers | `plot_histogram_with_fits`, `plot_qq_plots`, `plot_tail_comparison`, `plot_rolling_var` |
| `src/main.py` | Orchestration: ETL -> fit -> plots -> print VaR & CVaR for the selected `--methods` (Normal, Student-t, Historical, Bayes; Bayes saves `data/output/{coin}-bayes-summary.csv`) -> optional backtest + summary + rolling plot -> `plt.show()` | `etl(...)`, `main(...)`, `parse_methods(...)` |
| `tests/` | pytest: risk metric sign/order sanity, backtest test statistics (hand-checked values, edge cases), Student-t guards, Bayes (tiny seeded MCMC: shapes, nu > 2, finite VaR/CVaR, CVaR < VaR, same-seed reproducibility, exact predictive vs closed form and vs simulation, HDI, diagnostic warnings, `--methods` parsing) | 37 tests (1 marked `slow`, registered in `pytest.ini`) |

Known quirks (documented, not yet changed): the full-sample Normal fit uses ddof=0 while the rolling Normal uses ddof=1; `main.py` labels the risk-metric header `-USD` regardless of `--vs_currency`; the full-sample Historical VaR/CVaR is printed by `main.py` only with `--methods ...,hist` (default `normal,t` keeps the old output); `--methods` does not change the backtest methods.

## Stack (pinned in `requirements.txt`)
Python 3 with:

| Package | Version | Used for |
|---|---|---|
| numpy | 2.5.4 | arrays, log returns, percentiles |
| pandas | 3.1.0 | data frames, CSV I/O, rolling stats |
| scipy | 1.18.1 | `norm`, `t`, `t.fit`, `chi2`, `probplot` |
| yfinance | 1.7.0 | Yahoo Finance download |
| matplotlib | 3.11.2 | plots |
| seaborn | 0.13.2 | histogram |
| tqdm | 4.70.1 | backtest progress bar |
| pytest | 9.1.1 | tests |
| pymc | 6.3.2 | Bayesian Student-t (`src/bayes.py`, imported only with `--methods bayes`) |
| arviz | 1.3.0 | R-hat and bulk / tail ESS (`src/bayes.diagnostics`) |

## Related repos
| Repo | Relationship |
|---|---|
| `jvick1/crypto-risk-engine` (this repo) | Active repo; all new work lands here on feature branches. |
| `jvick1/student_t_var_cvar` | Superseded parent: the original Student-t VaR/CVaR code this repo grew from. No new work. |
| `jvick1/btc_alpha_engine` | Related stub (private; empty `main.py` and a short README). No code dependency in either direction. |

No runtime dependency on any other repo.

## Interfaces
### Main CLI: `python -m src.main`
| Flag | Type / default | Meaning |
|---|---|---|
| `--coin_symbol` | str, `btc` | Coin; upper-cased into the Yahoo ticker and used in file names |
| `--vs_currency` | str, `usd` | Quote currency; ticker is `{COIN}-{VS}` (e.g. `BTC-USD`) |
| `--alpha` | float, `0.05` | Tail probability (0.05 = 95% VaR), not a confidence level |
| `--backtest` | flag, off | Run the rolling backtest, save results + summary CSVs, show the rolling VaR plot |
| `--window_size` | int, `252` | Rolling window length in observations (crypto trades daily, so about 8 months) |
| `--methods` | comma list, `normal,t` | Full-sample methods to print: `normal`, `t`, `hist`, `bayes`. Bayes is opt-in (MCMC). Does not affect `--backtest` |
| `--seed` | int, `42` | Bayes: sampler seed (same seed + same input = same posterior) |
| `--draws` | int, `1000` | Bayes: posterior draws per chain |
| `--tune` | int, `1000` | Bayes: warm-up steps per chain (discarded) |
| `--chains` | int, `4` | Bayes: chains, run in parallel processes up to the CPU count |
| `--hdi_prob` | float, `0.94` | Bayes: credible-interval (HDI) mass |

### Standalone module CLIs (debugging)
- `python -m src.api --coin_symbol --vs_currency`: download and save raw CSV only.
- `python -m src.returns --coin_symbol --vs_currency [--verbose | --no-verbose]`: raw -> log returns (requires the raw CSV).
- `python -m src.distributions`, `python -m src.risk_metrics`, `python -m src.visualization` (`--coin_symbol`, `--vs_currency`): read the existing returns CSV and print fits / risk metrics (alpha fixed at 0.05) / show plots.
- `python -m src.backtest --coin_symbol --vs_currency --window_size --alpha`: backtest the existing returns CSV and print results; does **not** save CSVs.
- `python -m src.bayes --coin_symbol --alpha --draws --tune --chains --seed`: Bayesian Student-t on the existing returns CSV; prints only, does **not** save the CSV.

### Python API
`rolling_backtest(df_returns, window_size=252, alpha=0.05, methods=["normal","student_t","historical"])` expects a frame with `date` and `return` columns and returns `(results_df, summary_df)`; column layouts are in [SCHEMA.md](SCHEMA.md).

`run_bayes(returns, alpha=0.05, draws=1000, tune=1000, chains=4, seed=42, hdi_prob=0.94)` -> `(summary_df, runtime_seconds)`; layout in [SCHEMA.md](SCHEMA.md#5-bayes-summary-dataoutputcoin-bayes-summarycsv).

## Bayesian Student-t (Phase 3, `--methods bayes`)
Model on the full sample, sampled on `x / s` (s = sample std of returns, scale only, no centering) so NUTS sees order-1 parameters:

| Parameter | Prior (sampled scale) | Raw-scale meaning | Why |
|---|---|---|---|
| `mu` | `mu_z ~ Normal(0, 0.5)` | `mu ~ Normal(0, 0.5 s)` | Daily drift is tiny next to daily volatility; half a std allows any plausible drift without letting the center move into the tails |
| `sigma` | `sigma_z ~ HalfNormal(1)` | `sigma ~ HalfNormal(s)` | The t scale is below the std (`std = sigma * sqrt(nu / (nu - 2))`), so a HalfNormal at 1 std covers sensible values without being tight |
| `nu` | `nu = 2 + nu_minus_2`, `nu_minus_2 ~ Gamma(2, 0.1)` | prior mean 22 | nu > 2 keeps the variance finite (same reason as `DF_FLOOR`) without a hard wall; Gamma(2, 0.1) is a standard weakly informative prior for t degrees of freedom (Juarez & Steel 2010), covering very heavy tails through near-Normal |

Likelihood: `x_t / s ~ StudentT(nu, mu_z, sigma_z)`, i.i.d. (no volatility clustering, same as the MLE fit).

VaR / CVaR, two views, both reported:
- **Per draw (parameter uncertainty):** closed-form t VaR / CVaR for each posterior draw; the CSV reports mean, sd, median and the HDI (`--hdi_prob`, default 94%). The HDI says how sure we are about the risk number, given the model.
- **Posterior predictive (point estimate):** the predictive distribution of the next return is the mixture of the per-draw t's. Its alpha-quantile (VaR) and tail mean (CVaR) are computed exactly by root-finding and the closed-form t partial expectation. A simulated tail mean would be very noisy here because nu is near 2 (infinite fourth moment).

Diagnostics: rank-normalized R-hat and bulk / tail ESS (ArviZ) for mu, sigma, nu and the divergence count; a `RuntimeWarning` fires if any R-hat > 1.01, any ESS < 400 (about 100 per chain for 4 chains) or any divergence. With fewer chains or draws, low-ESS warnings are expected. The full-sample fit is not used by `--backtest` (Phase 4).

## Non-functional requirements
- **Logging / verbosity:** plain `print` output, a `tqdm` progress bar for the backtest, and `warnings.warn(RuntimeWarning)` on a Student-t fallback. No `logging` module. `main.py` always prints return summary stats (`verbose=True`); only `src.returns` exposes `--no-verbose`.
- **External dependency:** Yahoo Finance via `yfinance`, no API key. Every `main` run needs internet and re-downloads full history; an empty response raises `ValueError`. Yahoo data can be revised or rate-limited, so numbers drift slightly between runs.
- **Side effects:** each run overwrites the committed CSVs under `data/`; restore with `git checkout -- data` before committing.
- **Performance:** the full-sample run takes seconds (about 2 s on the box); `--methods bayes` adds about 5 s for BTC (4,439 returns, 4 chains x 2,000 iterations on 4 of 8 cores; 2026-11-12 run). The BTC rolling backtest (about 4,180 windows, one Student-t MLE per window) takes about 2 minutes. Faster backtest is backlog.
- **Reproducibility:** deterministic given the same input CSV (MLE fits, no random sampling). Tests use seeded synthetic data. Bayesian sampling is seeded by `--seed` (verified: two runs on the same CSV give identical posteriors); the posterior predictive VaR/CVaR is computed exactly, so it adds no extra randomness.
- **Security:** no secrets, credentials, API keys or personal data in the repo or required at runtime.
- **CI:** none yet; tests run locally with `python -m pytest`. GitHub Actions is backlog.
- **Statistical caveat:** rolling windows overlap, so the Kupiec/Christoffersen independence assumption is approximate; treat p-values as approximate.
