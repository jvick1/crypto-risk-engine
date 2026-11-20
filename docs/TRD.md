# Technical Requirements: Crypto Risk Engine

> Educational project, not investment advice. Describes the code on `feat/phase4-rolling-bayes-backtest`. Planned items are marked **Planned**.

## Architecture (by module)
A flat Python package `src/`, run as `python -m src.main`. Each module also has a standalone `__main__` for debugging.

| Module | Responsibility | Key functions |
|---|---|---|
| `src/api.py` | Extract: download daily data from Yahoo Finance via `yfinance` (`period="max"` or a `start` date, `interval="1d"`, `auto_adjust=False`), flatten MultiIndex columns, keep `Close` as `price`, save raw CSV. Optional `--cache_data` delta fetch (last date minus 5 days, merge with newer rows winning); `--refresh` forces a full download; Yahoo failures fall back to the cache with a warning when `--cache_data` is set | `fetch_historical_data(ticker, start=None)`, `read_raw_csv`, `merge_raw_prices`, `load_or_fetch_raw`, `save_raw_data` |
| `src/returns.py` | Transform: parse UTC timestamps, sort, drop duplicate dates, compute `log(p_t / p_{t-1})`, drop first NaN, save `date,return` | `compute_log_returns(...)`, `print_return_summary(r)` |
| `src/distributions.py` | Fit Normal (`mean`, `np.std` i.e. ddof=0) and Student-t (`scipy.stats.t.fit`) with df floor `DF_FLOOR = 2.01` (refit with `f0=2.01` if the MLE df is lower) and Normal-equivalent fallback (`DF_NORMAL_FALLBACK = 1e6`) plus a `RuntimeWarning` if the fit fails. Optional `start=(df, loc, scale)` seeds `t.fit` (used by `--t_warm_start`) | `fit_normal`, `fit_student_t` |
| `src/risk_metrics.py` | VaR / CVaR (left-tail quantiles of log returns, negative = loss) for Normal, Student-t (closed-form ES; raises `ValueError` for df <= 1, bad scale or alpha) and Historical (`np.percentile`, mean of returns <= VaR) | `compute_var_*`, `compute_cvar_*` |
| `src/backtest.py` | Rolling window backtest: for each day `i >= window_size`, fit on returns `[i-window_size, i)` and forecast day `i`; breach = `actual_return < VaR`. Normal uses pandas rolling mean/std (ddof=1); Student-t refits each window; Historical uses the window's empirical quantile; `bayes` (Phase 4) runs after that loop via `rolling_bayes`, so the other columns are unchanged. The MLE loop is parallelized with a chunked `ProcessPoolExecutor` (`spawn` start method, Windows-safe) when `workers>1`; `--workers 1` is the original serial path and is byte-identical. `--t_warm_start` seeds each t fit from the previous window and forces serial. Also a CLI on the existing returns CSV (`--save` writes the CSVs) | `rolling_backtest(..., workers=1, t_warm_start=False)`, `default_workers()`, `save_backtest_outputs(...)` |
| `src/rolling_bayes.py` | Rolling Bayesian VaR / CVaR (Phase 4): refit schedule every k days, conjugate Normal-Inverse-Gamma engine (closed form) and MCMC Student-t engine (Phase 3 priors, compiled once, warm-started), per-refit diagnostics, on-disk fit cache | `refit_schedule`, `nig_posterior`, `nig_predictive`, `rolling_bayes`, `cache_key`, `refit_report` |
| `src/backtest_tests.py` | Kupiec POF (chi2, 1 df), Christoffersen independence (chi2, 1 df), conditional coverage `LR_cc = LR_pof + LR_ind` (chi2, 2 df), reject flags at 5%; `0*log(0)=0` for edge cases | `kupiec_pof`, `christoffersen_independence`, `christoffersen_cc` |
| `src/bayes.py` | Bayesian Student-t on the full sample (PyMC NUTS). Samples on `x / s` (s = sample std, no centering) with priors `mu_z ~ Normal(0, 0.5)`, `sigma_z ~ HalfNormal(1)`, `nu = 2 + Gamma(2, 0.1)`, i.e. `mu ~ Normal(0, 0.5 s)`, `sigma ~ HalfNormal(s)` on the raw scale. Per posterior draw: closed-form t VaR / CVaR (reuses `risk_metrics`), summarized by mean, sd, median and HDI (own narrowest-interval `hdi`, default 94%). Posterior predictive VaR / CVaR computed exactly from the mixture of per-draw t's (`brentq` on the mixture CDF + closed-form t partial expectation; no simulation noise). Diagnostics: ArviZ rank-normalized R-hat and bulk / tail ESS for mu, sigma, nu, plus divergence count; `RuntimeWarning` if R-hat > `RHAT_MAX` (1.01), ESS < `ESS_MIN` (400) or any divergence | `build_model`, `fit_bayes_student_t`, `summarize`, `predictive_var_cvar`, `diagnostics`, `check_diagnostics`, `run_bayes`, `print_bayes_summary` |
| `src/visualization.py` | Matplotlib/seaborn figures: histogram + fitted PDFs with VaR lines, QQ plots, log-scale left-tail density, rolling VaR with breach markers | `plot_histogram_with_fits`, `plot_qq_plots`, `plot_tail_comparison`, `plot_rolling_var` |
| `src/timing.py` | Nestable `stage(name)` context manager (`time.perf_counter` + `logging`), run-level collector, `--profile` CSV, optional `on_stage` progress hook | `stage`, `TimingCollector`, `reset_collector`, `configure_logging` |
| `src/main.py` | Orchestration: ETL -> fit -> plots -> print VaR & CVaR for the selected `--methods` (Normal, Student-t, Historical, Bayes; Bayes saves `data/output/{coin}-bayes-summary.csv`) -> optional backtest + summary + rolling plot -> `plt.show()` (skipped with `--no_plots`); `--profile` writes the timings CSV | `etl(...)`, `main(...)`, `parse_methods(...)`, `build_parser()` |
| `tests/` | pytest: risk metric sign/order sanity, backtest test statistics (hand-checked values, edge cases), Student-t guards, Bayes (tiny seeded MCMC: shapes, nu > 2, finite VaR/CVaR, CVaR < VaR, same-seed reproducibility, exact predictive vs closed form and vs simulation, HDI, diagnostic warnings, `--methods` parsing including the CLI default), rolling Bayes (refit schedule, no look-ahead under perturbed future data, NIG posterior vs a hand calculation and vs simulation, existing backtest columns unchanged, cache hit / miss / invalidation with a fake sampler, warm-start inits, diagnostic failure counting, one tiny real-MCMC test), timing collector (nesting, CSV columns, progress hook order), parallel == serial on a small series plus the `on_stage` hook under `workers=2`, `--t_warm_start` runs, raw-data cache merge / refresh / rate-limit fallback (mocked fetch, no network), yfinance MultiIndex flatten | 79 tests (2 marked `slow`, registered in `pytest.ini`) |

Known quirks (documented, not yet changed): the full-sample Normal fit uses ddof=0 while the rolling Normal uses ddof=1; the standalone `python -m src.backtest --methods` names are `normal,student_t,historical`, not the `normal,t,hist` names used by `src.main`; the full-sample Historical VaR/CVaR is printed by `main.py` only with `--methods ...,hist` (default `normal,t` keeps the old output); `--backtest` always runs Normal, Student-t and Historical; `--methods` only adds the rolling Bayes backtest when it lists `bayes`.

## Stack (minimum versions in `requirements.txt` and `environment.yml`)
Python 3.11 or newer. Tested on Python 3.11 (PyMC 5.28 / ArviZ 0.23) and 3.12 (PyMC 6.3 / ArviZ 1.3). Anaconda users should create the separate env in `environment.yml` rather than installing into base.

| Package | Minimum | Used for |
|---|---|---|
| numpy | 1.26 | arrays, log returns, percentiles |
| pandas | 2.1 | data frames, CSV I/O, rolling stats |
| scipy | 1.11 | `norm`, `t`, `t.fit`, `chi2`, `probplot` |
| yfinance | 0.2.40 | Yahoo Finance download |
| matplotlib | 3.8 | plots |
| seaborn | 0.13 | histogram |
| tqdm | 4.66 | backtest progress bar |
| pytest | 8 | tests |
| pymc | 5.10 | Bayesian Student-t (`src/bayes.py`, `src/rolling_bayes.py` MCMC engine; imported only with `--methods bayes`) |
| arviz | 0.17 | R-hat and bulk / tail ESS (`src/bayes.diagnostics`) |

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
| `--methods` | comma list, `normal,t` | Full-sample methods to print: `normal`, `t`, `hist`, `bayes`. Bayes is opt-in (MCMC). `--backtest` always runs Normal, Student-t and Historical, and adds the rolling Bayes backtest when `bayes` is listed |
| `--seed` | int, `42` | Bayes: sampler seed (same seed + same input = same posterior) |
| `--draws` | int, `1000` | Bayes: posterior draws per chain |
| `--tune` | int, `1000` | Bayes: warm-up steps per chain (discarded) |
| `--chains` | int, `4` | Bayes: chains, run in parallel processes up to the CPU count |
| `--hdi_prob` | float, `0.94` | Bayes: credible-interval (HDI) mass |
| `--bayes_engine` | `mcmc` / `conjugate`, `mcmc` | Backtest Bayes engine (only with `--backtest` and `bayes` in `--methods`) |
| `--refit_every` | int, `21` (mcmc) / `1` (conjugate) | Backtest Bayes: days between refits (k) |
| `--bt_draws`, `--bt_tune`, `--bt_chains` | int, `500`, `500`, `2` | Backtest Bayes (mcmc): NUTS draws / tuning steps / chains per refit (chains run sequentially in one process) |
| `--no_cache` | flag, off | Backtest Bayes (mcmc): do not read or write `data/cache/rolling-bayes/` |
| `--log-level` | str, `WARNING` | Python logging level; `INFO` shows per-stage start/end lines from the `crypto_risk` logger |
| `--profile` | flag, off | Print a stage timing table and write `data/output/{coin}-timings.csv` (gitignored) |
| `--no_plots` | flag, off | Skip plot generation and `plt.show()` |
| `--workers` | int, `min(4, cpu_count)` | Process count for the rolling Normal / Student-t / Historical loop. `--workers 1` is the serial path (byte-identical output). Library `rolling_backtest()` defaults to 1 |
| `--t_warm_start` | flag, off | Seed each Student-t window fit with the previous window's (df, loc, scale). Forces serial. Default off so outputs stay identical to a cold fit |
| `--cache_data` | flag, off | If `data/raw/{coin}-{vs}-max.csv` exists, re-fetch from last date minus 5 days and merge (newer rows win). Yahoo failures fall back to the cache with a warning |
| `--refresh` | flag, off | Force a full `period=max` download, even with `--cache_data` |

`--seed` also seeds the rolling MCMC refits (refit on day `r` uses `seed + r`).

### Standalone module CLIs (debugging)
- `python -m src.api --coin_symbol --vs_currency [--cache_data] [--refresh]`: download and save raw CSV only.
- `python -m src.returns --coin_symbol --vs_currency [--verbose | --no-verbose]`: raw -> log returns (requires the raw CSV).
- `python -m src.distributions`, `python -m src.risk_metrics`, `python -m src.visualization` (`--coin_symbol`, `--vs_currency`): read the existing returns CSV and print fits / risk metrics (alpha fixed at 0.05) / show plots.
- `python -m src.backtest --coin_symbol --vs_currency --window_size --alpha [--methods normal,student_t,historical,bayes] [--bayes_engine] [--refit_every] [--bt_draws --bt_tune --bt_chains] [--seed] [--no_cache] [--workers] [--t_warm_start] [--save]`: backtest the existing returns CSV (no download) and print results and the runtime; writes the results / summary / refit CSVs only with `--save`.
- `python -m src.bayes --coin_symbol --alpha --draws --tune --chains --seed`: Bayesian Student-t on the existing returns CSV; prints only, does **not** save the CSV.

### Python API
`rolling_backtest(df_returns, window_size=252, alpha=0.05, methods=["normal","student_t","historical"], bayes_engine="mcmc", refit_every=None, bayes_draws=500, bayes_tune=500, bayes_chains=2, seed=42, cache_dir=None, return_refits=False, workers=1, t_warm_start=False)` expects a frame with `date` and `return` columns and returns `(results_df, summary_df)`, or `(results_df, summary_df, refits_df)` with `return_refits=True` (`refits_df` is `None` without `bayes`). The MCMC fit cache is off unless `cache_dir` is given (the CLIs pass `data/cache/rolling-bayes`). `workers=1` is the serial path; `workers>1` uses a chunked process pool (`spawn`) and is byte-identical. `t_warm_start=True` with `workers>1` warns and forces serial. Column layouts are in [SCHEMA.md](SCHEMA.md).

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

Diagnostics: rank-normalized R-hat and bulk / tail ESS (ArviZ) for mu, sigma, nu and the divergence count; a `RuntimeWarning` fires if any R-hat > 1.01, any ESS < 400 (about 100 per chain for 4 chains) or any divergence. With fewer chains or draws, low-ESS warnings are expected. The full-sample fit is not used by `--backtest` (that would be look-ahead); the rolling backtest refits per window (next section).

## Rolling Bayesian backtest (Phase 4, `--backtest --methods ...,bayes`)
Code: `src/rolling_bayes.py`, called from `rolling_backtest` after the existing per-day loop, so the Normal / Student-t / Historical columns are byte-for-byte the same as before (checked by diffing the BTC CSV against `main`, and by a unit test).

**Refit schedule and no look-ahead.** Forecast days are positions `i = window_size .. n-1` in the date-sorted returns. Refits happen on days `r = window_size, window_size + k, window_size + 2k, ...` (`k = --refit_every`). The refit on day `r` uses only returns `[r - window_size, r)`, i.e. data that ends the day before `r`, and its posterior gives the forecast for every day in `[r, r + k)`. So the forecast for day `i` never uses returns at or after `i`; with `k = 1` it uses exactly `[i - window_size, i)` like the other methods. The schedule is anchored at the first forecast day, so appending new data keeps every earlier refit (and its cache entry). Tests wreck the future returns and check that past forecasts are unchanged.

**Default k.** `mcmc`: 21 (about one trading month). A 252-day window changes by only 21/252 (8%) of its data between refits, so the posterior moves little, and about 190 refits keep a full BTC backtest to minutes. `conjugate`: 1 (daily), because a refit is a few arithmetic operations.

**MCMC engine (default).** The Phase 3 Student-t with the same priors (`mu_z ~ Normal(0, 0.5)`, `sigma_z ~ HalfNormal(1)`, `nu = 2 + Gamma(2, 0.1)`, on `x / s` with `s` the window std). The model is compiled once with `pm.Data` and each window is swapped in with `pm.set_data`. Per refit: `--bt_chains 2` x `--bt_draws 500` after `--bt_tune 500` (smaller than the full-sample 4 x 1,000 because each window has 252 points, not 4,000+). VaR / CVaR are the exact posterior predictive values (`bayes.predictive_var_cvar`), as in Phase 3.
- **Warm start:** each refit's NUTS starts at the previous refit's posterior means of mu, sigma, nu (mapped to the new window's scale, rounded to 6 significant digits). The priors are **not** replaced by the previous posterior: consecutive windows share `window_size - k` days, so that would count those days twice. Each window's posterior depends only on its own data and the fixed priors; warm start only shortens the sampler's start.
- **Diagnostics per refit:** R-hat and bulk / tail ESS (ArviZ) for mu, sigma, nu plus divergences. A refit fails if any R-hat > 1.01, any ESS < 100 per chain (200 for 2 chains) or any divergence. Failures are recorded in the refit log (`passed`, `problems`) and counted in one console line; per-refit warnings are suppressed. A failed refit's VaR is still used (it is flagged, not dropped).
- **Seeds:** refit on day `r` uses `--seed + r`, so results are reproducible and do not depend on k's numbering.

**Conjugate engine (`--bayes_engine conjugate`).** Normal likelihood `x ~ Normal(mu, sigma^2)` with a Normal-Inverse-Gamma prior `sigma^2 ~ InvGamma(a0, b0)`, `mu | sigma^2 ~ Normal(m0, sigma^2 / kappa0)`. Posterior for a window of n returns with mean `xbar` and `SS = sum((x - xbar)^2)`: `kappa_n = kappa0 + n`, `m_n = (kappa0 m0 + n xbar) / kappa_n`, `a_n = a0 + n/2`, `b_n = b0 + SS/2 + kappa0 n (xbar - m0)^2 / (2 kappa_n)`. The posterior predictive of the next return is a Student-t with `df = 2 a_n`, location `m_n`, scale `sqrt(b_n (kappa_n + 1) / (a_n kappa_n))`, so VaR / CVaR use the closed-form t formulas in `risk_metrics`.
- **Priors:** `m0 = 0` (no drift), `kappa0 = 1` (worth one observation for the mean), `a0 = 1`, `b0 = 1e-4` (worth two observations of variance with a 1% daily-volatility guess). With n = 252 the data dominate (prior weight under 1%).
- **Limitation:** the likelihood is Normal, so the predictive t's fat tails come **only** from parameter uncertainty. With 252 observations `df = 2 a_n = 254`, so the conjugate VaR is essentially a Normal VaR with a slightly wider scale; it does not model crypto's fat tails. It is a fast baseline and a check on the MCMC engine, not a fat-tail model.
- Not cached: all ~3,900 daily refits for BTC take well under a second.

**Cache (MCMC).** Each refit's draws and diagnostics are stored in `data/cache/rolling-bayes/{sha256}.npz` (gitignored). The key hashes the exact float64 bytes of the window, the model / prior tag, a cache version, draws, tune, chains, the per-refit seed and the warm-start init values, so a change to any input, prior, sampler setting or the init chain is a miss. Including the init means a different k (different init chain) does not reuse fits, which keeps results bit-identical to an uncached run. `alpha` is not in the key: VaR / CVaR are recomputed from the cached draws. Corrupt files count as misses; writes are atomic. With all refits cached, PyMC is never imported. `--no_cache` turns it off.

**Outputs.** `bayes_var`, `bayes_cvar`, `bayes_breach` in the results CSV; a `bayes` row in the summary CSV from the same `backtest_tests.christoffersen_cc`; the per-refit log `data/backtest/{coin}-bayes-refits.csv`. See [SCHEMA.md](SCHEMA.md).

**Caveats.**
- Overlapping windows: consecutive forecasts share most of their data, so breaches are not independent and the Kupiec / Christoffersen p-values are approximate (same as the other methods).
- Stale posterior between refits: with `k > 1` the forecast for day `i` uses a fit that ends up to `k` days before `i`, so it reacts to volatility changes up to k - 1 days late; this is a fair, no-look-ahead comparison but not the same information set as the daily-refit methods.
- Conjugate Normal likelihood: tails come only from parameter uncertainty (see above).
- Both engines assume i.i.d. returns within a window (no volatility clustering), like the other methods.
- Runtime numbers are wall-clock on the 8-core box and vary with load.

## Non-functional requirements
- **Logging / verbosity:** plain `print` output, a `tqdm` progress bar for the backtest (and one for rolling Bayes refits; PyMC's per-refit log lines are silenced), Python `logging` via `--log-level` (default `WARNING`; `INFO` shows per-stage start/end times from the `crypto_risk` logger), and `warnings.warn(RuntimeWarning)` on a Student-t fallback, a forced-serial warm start, or a cache-data Yahoo fallback. `main.py` always prints return summary stats (`verbose=True`); only `src.returns` exposes `--no-verbose`.
- **External dependency:** Yahoo Finance via `yfinance`, no API key. By default every `main` run needs internet and re-downloads full history; an empty full-history response raises `ValueError`. With `--cache_data`, only the recent delta is fetched and a Yahoo failure or rate limit falls back to the cached raw CSV with a warning. Yahoo data can be revised or rate-limited, so numbers can still drift between full-download runs.
- **Side effects:** each run overwrites the committed CSVs under `data/`; restore with `git checkout -- data` before committing.
- **Performance:** Phase 6 serial baseline (live BTC, 2026-11-20, alpha 0.05, window 252, 4,447 returns, 4,195 forecast days): conjugate `--profile --no_plots` was 113.102 s top-level, of which the Student-t MLE `backtest loop` was 106.658 s; see the ranked table in [APP_FLOW.md](APP_FLOW.md). Phase 7 (same day, committed BTC returns 2014-09-18 to 2026-02-28, 4,182 returns, 3,930 forecast days, 8 CPUs): the `backtest loop` is 102.237 s with `--workers 1` and 28.016 s with the CLI default of 4 workers (~3.65x, byte-identical). Optional `--t_warm_start` is 36.268 s serial but changes Student-t numbers (documented in [ROADMAP.md](ROADMAP.md)). A `--cache_data` delta fetch onto the committed raw file was 0.101 s vs 0.188 s for a full `period=max` download (4,448 raw rows through 2026-11-20; network-dependent).
- **Reproducibility:** deterministic given the same input CSV (MLE fits, no random sampling; the rolling MCMC refits are seeded per refit and the cache stores exactly what an uncached run would produce). Tests use seeded synthetic data. Bayesian sampling is seeded by `--seed` (verified: two runs on the same CSV give identical posteriors); the posterior predictive VaR/CVaR is computed exactly, so it adds no extra randomness.
- **Security:** no secrets, credentials, API keys or personal data in the repo or required at runtime.
- **CI:** none yet; tests run locally with `python -m pytest`. GitHub Actions is backlog.
- **Statistical caveat:** rolling windows overlap, so the Kupiec/Christoffersen independence assumption is approximate; treat p-values as approximate.
