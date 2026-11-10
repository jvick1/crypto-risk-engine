# Technical Requirements: Crypto Risk Engine

> Educational project, not investment advice. Describes the code on `feat/phase2-student-t-guards`. Planned items are marked **Planned**.

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
| `src/visualization.py` | Matplotlib/seaborn figures: histogram + fitted PDFs with VaR lines, QQ plots, log-scale left-tail density, rolling VaR with breach markers | `plot_histogram_with_fits`, `plot_qq_plots`, `plot_tail_comparison`, `plot_rolling_var` |
| `src/main.py` | Orchestration: ETL -> fit -> plots -> print Normal/Student-t VaR & CVaR -> optional backtest + summary + rolling plot -> `plt.show()` | `etl(...)`, `main(...)` |
| `tests/` | pytest: risk metric sign/order sanity, backtest test statistics (hand-checked values, edge cases), Student-t guards | 26 tests |

Known quirks (documented, not yet changed): the full-sample Normal fit uses ddof=0 while the rolling Normal uses ddof=1; `main.py` labels the risk-metric header `-USD` regardless of `--vs_currency`; the full-sample Historical VaR/CVaR is computed only by `python -m src.risk_metrics`, not printed by `main.py`.

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
| pymc | 6.3.2 | **Planned** (Phase 3 Bayesian model); pinned, not imported yet |
| arviz | 1.3.0 | **Planned** (Phase 3 R-hat / ESS diagnostics); pinned, not imported yet |

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

### Standalone module CLIs (debugging)
- `python -m src.api --coin_symbol --vs_currency`: download and save raw CSV only.
- `python -m src.returns --coin_symbol --vs_currency [--verbose | --no-verbose]`: raw -> log returns (requires the raw CSV).
- `python -m src.distributions`, `python -m src.risk_metrics`, `python -m src.visualization` (`--coin_symbol`, `--vs_currency`): read the existing returns CSV and print fits / risk metrics (alpha fixed at 0.05) / show plots.
- `python -m src.backtest --coin_symbol --vs_currency --window_size --alpha`: backtest the existing returns CSV and print results; does **not** save CSVs.

### Python API
`rolling_backtest(df_returns, window_size=252, alpha=0.05, methods=["normal","student_t","historical"])` expects a frame with `date` and `return` columns and returns `(results_df, summary_df)`; column layouts are in [SCHEMA.md](SCHEMA.md).

### Planned interface (Phase 3+)
`--methods` flag including `bayes` (PyMC Student-t with nu > 2, posterior-predictive VaR/CVaR with credible intervals, R-hat/ESS). **Planned, not built.**

## Non-functional requirements
- **Logging / verbosity:** plain `print` output, a `tqdm` progress bar for the backtest, and `warnings.warn(RuntimeWarning)` on a Student-t fallback. No `logging` module. `main.py` always prints return summary stats (`verbose=True`); only `src.returns` exposes `--no-verbose`.
- **External dependency:** Yahoo Finance via `yfinance`, no API key. Every `main` run needs internet and re-downloads full history; an empty response raises `ValueError`. Yahoo data can be revised or rate-limited, so numbers drift slightly between runs.
- **Side effects:** each run overwrites the committed CSVs under `data/`; restore with `git checkout -- data` before committing.
- **Performance:** the full-sample run takes seconds; the BTC rolling backtest (about 4,180 windows, one Student-t MLE per window) takes about 2 minutes. Faster backtest is backlog.
- **Reproducibility:** deterministic given the same input CSV (MLE fits, no random sampling). Tests use seeded synthetic data. Phase 3 Bayesian sampling must be seeded (**Planned**).
- **Security:** no secrets, credentials, API keys or personal data in the repo or required at runtime.
- **CI:** none yet; tests run locally with `python -m pytest`. GitHub Actions is backlog.
- **Statistical caveat:** rolling windows overlap, so the Kupiec/Christoffersen independence assumption is approximate; treat p-values as approximate.
