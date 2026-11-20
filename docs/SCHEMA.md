# Data Schema: Crypto Risk Engine

> Educational project, not investment advice. All storage is local CSV under `data/`; there is no database, JSON feed, or config file. `{coin}` and `{vs}` are the `--coin_symbol` / `--vs_currency` values exactly as typed (lower case by convention).

## Files

### 1. Raw prices: `data/raw/{coin}-{vs}-max.csv`
Written by `api.save_raw_data`. One row per Yahoo daily bar, full history (`period="max"`).

| Column | Type | Meaning |
|---|---|---|
| `snapped_at` | datetime, UTC, ISO text (e.g. `2014-09-17 00:00:00+00:00`) | Bar date |
| `price` | float, rounded to 8 dp | Daily `Close` (not adjusted; `auto_adjust=False`) |

Committed examples: `btc-usd`, `eth-usd`, `sol-usd`, `doge-usd`, `xrp-usd` (ascending by date).

### 2. Log returns: `data/output/{coin}-log-returns.csv`
Written by `returns.compute_log_returns`. Sorted, de-duplicated by date; first row dropped (no prior price).

| Column | Type | Meaning |
|---|---|---|
| `date` | date `YYYY-MM-DD` | Date of the return (UTC calendar day) |
| `return` | float | `ln(price_t / price_{t-1})` |

Current code writes ascending dates. Note: the committed `eth`, `doge` and `xrp` files are in **descending** order from an older version of the code; regenerate them with `python -m src.main --coin_symbol eth` etc. `rolling_backtest` re-sorts by date, so either order is safe for the backtest.

### 3. Backtest results: `data/backtest/{coin}-backtest-results.csv`
Written by `main.py` with `--backtest` (and by `python -m src.backtest --save`). One row per out-of-sample day (`n - window_size` rows). `{m}` is each of `normal`, `student_t`, `historical`, plus `bayes` when `bayes` is in `--methods` (Phase 4).

| Column | Type | Meaning |
|---|---|---|
| `date` | date | Forecast day |
| `actual_return` | float | Realized log return that day |
| `{m}_var` | float (negative = loss) | VaR at `alpha` from the prior `window_size` returns (for `bayes`: from the window ending the day before the last refit, see below) |
| `{m}_cvar` | float (negative = loss) | CVaR / expected shortfall at `alpha` |
| `{m}_breach` | bool | `actual_return < {m}_var` |

Full column order: `date, actual_return, normal_var, normal_cvar, normal_breach, student_t_var, student_t_cvar, student_t_breach, historical_var, historical_cvar, historical_breach`, then `bayes_var, bayes_cvar, bayes_breach` when Bayes runs. The first 11 columns are identical with or without Bayes.

`bayes_var` / `bayes_cvar` are the posterior predictive VaR / CVaR of the engine chosen with `--bayes_engine` (the column names do not change with the engine; the refit log below records it). With `--refit_every k`, the value for day `i` comes from the refit on day `r = window_size + floor((i - window_size) / k) * k <= i`, fitted on returns `[r - window_size, r)`, so it is constant within each k-day block.

Committed examples: `btc` (Phase 4 run on the committed `btc-log-returns.csv`, 2014-09-18 to 2026-02-28: 3,930 forecast days 2015-05-28 to 2026-02-28, alpha 0.05, window 252, Bayes = MCMC engine at k = 21, seed 42) and `sol` (from an earlier run, before the Phase 2 df floor, no Bayes columns).

### 4. Backtest summary: `data/backtest/{coin}-backtest-summary.csv`
Written by `main.py` with `--backtest` (added in Phase 1; none committed, matching the repo's convention of committing only the results CSV). One row per method.

| Column | Type | Meaning |
|---|---|---|
| `method` | str (index) | `normal`, `student_t`, `historical`, and `bayes` when run (Phase 4) |
| `breaches` | int | Count of breach days |
| `total_periods` | int | Out-of-sample days |
| `expected_breaches` | float | `alpha * total_periods` |
| `breach_rate` | float | `breaches / total_periods` |
| `expected_rate` | float | `alpha` |
| `lr_pof`, `p_pof` | float | Kupiec POF LR statistic and p-value (chi2, 1 df) |
| `lr_ind`, `p_ind` | float | Christoffersen independence LR and p-value (chi2, 1 df) |
| `lr_cc`, `p_cc` | float | Conditional coverage LR (`lr_pof + lr_ind`) and p-value (chi2, 2 df) |
| `reject_pof_5pct`, `reject_ind_5pct`, `reject_cc_5pct` | bool | p-value < 0.05 |

### 4b. Bayes refit log: `data/backtest/{coin}-bayes-refits.csv`
Written by `main.py` with `--backtest` and `bayes` in `--methods` (Phase 4; `src.rolling_bayes.rolling_bayes`). One row per refit. Not committed.

| Column | Type | Meaning |
|---|---|---|
| `engine` | str | `mcmc` or `conjugate` (`--bayes_engine`) |
| `refit_every` | int | k (`--refit_every`; default 21 for `mcmc`, 1 for `conjugate`) |
| `refit_date` | date | First forecast day that uses this fit (the refit day `r`) |
| `fit_start_date`, `fit_end_date` | date | First and last return in the fit window; `fit_end_date` is the day before `refit_date` |
| `refit_pos`, `fit_start_pos`, `fit_end_pos` | int | Same, as 0-based positions in the date-sorted returns |
| `n_obs` | int | Window length (`window_size`) |
| `mu_mean` | float | Posterior mean of the location (mcmc: t location; conjugate: NIG `m_n`) |
| `sigma_mean` | float | mcmc: posterior mean of the t scale; conjugate: `sqrt(E[sigma^2]) = sqrt(b_n / (a_n - 1))` |
| `nu_mean` | float | mcmc: posterior mean of nu; empty for conjugate |
| `pred_df`, `pred_loc`, `pred_scale` | float | conjugate only: the Student-t posterior predictive (`2 a_n`, `m_n`, `sqrt(b_n (kappa_n + 1) / (a_n kappa_n))`) |
| `r_hat_max`, `ess_bulk_min`, `ess_tail_min` | float | mcmc only: worst R-hat and smallest bulk / tail ESS over mu, sigma, nu |
| `divergences` | int | mcmc only: divergent transitions after tuning |
| `passed` | bool | mcmc only: R-hat <= 1.01, every ESS >= 100 per chain, and 0 divergences |
| `problems` | str | mcmc only: failed checks, comma-separated from `rhat`, `ess`, `divergences` (empty if passed) |
| `cached` | bool | mcmc only: fit loaded from the cache instead of sampled |
| `seed` | int | mcmc only: per-refit seed, `--seed + refit_pos` |
| `cache_key` | str | mcmc only: first 16 hex characters of the cache key |
| `var`, `cvar` | float | Predictive VaR / CVaR at `alpha` used for the next k days |
| `seconds` | float | Wall time for this refit (near 0 for cache hits) |

### 4c. Fit cache: `data/cache/rolling-bayes/{sha256}.npz` (gitignored)
Written by the MCMC engine unless `--no_cache`. One NumPy `.npz` per refit with the posterior draws `mu`, `sigma`, `nu` (chains x draws, flattened chain-major), `r_hat`, `ess_bulk`, `ess_tail` (for mu, sigma, nu) and `divergences`. The file name is a SHA-256 of the window's float64 bytes, the model / prior tag (`MODEL_TAG`), `CACHE_VERSION`, draws, tune, chains, the per-refit seed and the warm-start init values, so any change to these is a cache miss. `alpha` is not in the key (VaR / CVaR are recomputed from the draws). Corrupt files are treated as misses. Delete the folder to clear the cache. Writes are atomic (temp file + rename).

### 5. Bayes summary: `data/output/{coin}-bayes-summary.csv`
Written by `main.py` with `--methods ...,bayes` (added in Phase 3; `src.bayes.summarize`). Full sample, one row per quantity. VaR / CVaR are log-return quantiles at `alpha` (negative = loss), like the other methods.

| Column | Type | Meaning |
|---|---|---|
| `quantity` | str (index) | `mu`, `sigma`, `nu` (Student-t location, scale, degrees of freedom; `sigma` is the t scale, not the std), `var`, `cvar` |
| `mean`, `sd`, `median` | float | Posterior mean, sd (ddof=1) and median over all draws (for `var` / `cvar`: of the per-draw closed-form t VaR / CVaR) |
| `hdi_low`, `hdi_high` | float | Highest-density credible interval with mass `hdi_prob` |
| `predictive` | float | `var` / `cvar` rows only: exact posterior predictive VaR / CVaR (alpha-quantile and tail mean of the mixture of per-draw t's); empty for parameters |
| `r_hat` | float | Rank-normalized R-hat (parameter rows only; warn if > 1.01) |
| `ess_bulk`, `ess_tail` | float | Bulk and tail effective sample size (parameter rows only; warn if < 400) |
| `alpha` | float | Tail probability used |
| `hdi_prob` | float | HDI mass (default 0.94) |
| `n_obs` | int | Number of finite log returns fitted |
| `chains`, `draws` | int | Chains and kept draws per chain (`--tune` warm-up is discarded and not recorded) |
| `seed` | int | `--seed` used |
| `divergences` | int | Divergent NUTS transitions after tuning (should be 0) |

Committed example: `btc`, from the live 2026-11-12 run (alpha 0.05, `n_obs` 4,439, 4 chains x 1,000 draws, seed 42). It is newer than the committed `btc-log-returns.csv` (which ends 2026-02-28), so rerunning on the committed CSV with `python -m src.bayes` gives slightly different numbers.

### 6. Timing summary: `data/output/{coin}-timings.csv`
Written by `main.py` with `--profile` (`src/timing.py`). Gitignored (`data/output/*-timings.csv`). One row per pipeline stage, in start order (a parent appears before its children). `parent` is empty for top-level stages. Nested stages (`backtest loop`, `rolling bayes`, `backtest tests`) have parent `backtest`. Plot stages are omitted when `--no_plots` is set.

| Column | Type | Meaning |
|---|---|---|
| `run_id` | str | UTC timestamp plus 8 hex characters, e.g. `20261120T224639Z-f16ad198` |
| `stage` | str | Stage name, matching the node labels in [APP_FLOW.md](APP_FLOW.md): `data pull`, `returns`, `fits`, `risk metrics`, `bayes`, `plots`, `backtest`, `backtest loop`, `rolling bayes`, `backtest tests`, `rolling var plot` |
| `parent` | str | Parent stage name, or empty for top-level stages |
| `seconds` | float | Wall-clock time from `time.perf_counter` |
| `coin_symbol`, `vs_currency`, `alpha`, `backtest`, `window_size`, `methods`, `seed`, `draws`, `tune`, `chains`, `hdi_prob`, `bayes_engine`, `refit_every`, `bt_draws`, `bt_tune`, `bt_chains`, `use_cache`, `no_plots` | mixed | Run settings, repeated on every row (`methods` is the comma-separated `--methods` string; `refit_every` is empty when the default was left unset) |

## Plot outputs
Plot functions in `visualization.py` return matplotlib Figures; `main` shows them with `plt.show()` at the end of the run unless `--no_plots` is set (which skips generation and show). **They are not saved to disk** (the README images are hosted on GitHub).

| Figure | Function | When |
|---|---|---|
| Return histogram with Normal and Student-t PDFs and VaR lines | `plot_histogram_with_fits` | Every run except `--no_plots` |
| QQ plots vs Normal and vs Student-t | `plot_qq_plots` | Every run except `--no_plots` |
| Left-tail density, log scale (x from -0.30 to 0) | `plot_tail_comparison` | Every run except `--no_plots` |
| Rolling VaR per method with breach markers and breach rate in legend | `plot_rolling_var` | `--backtest` only, except `--no_plots` |

## Config
There is no config file or environment variable. Configuration is the CLI flags (see [TRD.md](TRD.md#interfaces)) plus these code constants:

| Constant | Value | Where |
|---|---|---|
| `DF_FLOOR` | `2.01` | `src/distributions.py` (minimum Student-t df) |
| `DF_NORMAL_FALLBACK` | `1e6` | `src/distributions.py` (df used when the t fit fails) |
| Test significance `sig` | `0.05` | `src/backtest_tests.christoffersen_cc` |
| Yahoo download | `period="max"`, `interval="1d"`, `auto_adjust=False` | `src/api.py` |
| Default methods | `["normal", "student_t", "historical"]` | `src/backtest.rolling_backtest` |
| Default `--methods` | `normal,t` (Bayes and Historical opt-in) | `src/main.DEFAULT_METHODS` |
| `RHAT_MAX` | `1.01` | `src/bayes.py` (R-hat warning threshold) |
| `ESS_MIN` | `400` | `src/bayes.py` (bulk / tail ESS warning threshold) |
| Bayes priors | `mu_z ~ Normal(0, 0.5)`, `sigma_z ~ HalfNormal(1)`, `nu = 2 + Gamma(2, 0.1)` on `returns / std` | `src/bayes.build_model` (rationale in [TRD.md](TRD.md#bayesian-student-t-phase-3---methods-bayes)) |

| Rolling Bayes refit defaults | `--refit_every` 21 (mcmc) / 1 (conjugate); `--bt_draws 500`, `--bt_tune 500`, `--bt_chains 2` (chains run sequentially) | `src/rolling_bayes.py` (`DEFAULT_REFIT_EVERY`, `BT_DRAWS`, `BT_TUNE`, `BT_CHAINS`) |
| Rolling Bayes diagnostic thresholds | R-hat <= 1.01, ESS >= 100 per chain, 0 divergences | `src/rolling_bayes.py` (`RHAT_MAX`, `ESS_MIN_PER_CHAIN`) |
| Conjugate NIG prior | `m0 = 0`, `kappa0 = 1`, `a0 = 1`, `b0 = 1e-4` | `src/rolling_bayes.py` (`NIG_M0`, `NIG_KAPPA0`, `NIG_A0`, `NIG_B0`) |
| Cache folder | `data/cache/rolling-bayes/` (gitignored; `--no_cache` disables) | `src/main.py`, `src/backtest.py` |
