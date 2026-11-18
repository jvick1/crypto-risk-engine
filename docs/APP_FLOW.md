# App Flow: Crypto Risk Engine

> Educational project, not investment advice. Planned steps are marked **Planned**. File layouts are in [SCHEMA.md](SCHEMA.md).

## User flow
1. Install: `python -m venv .venv`, activate, `pip install -r requirements.txt`.
2. Run `python -m src.main --coin_symbol btc` (add `--vs_currency`, `--alpha`, `--backtest`, `--window_size`, `--methods normal,t,hist,bayes` and the Bayes flags `--seed`, `--draws`, `--tune`, `--chains`, `--hdi_prob` as needed; for the rolling Bayes backtest add `--backtest` with `bayes` in `--methods`, plus `--bayes_engine`, `--refit_every`, `--bt_draws`, `--bt_tune`, `--bt_chains`, `--no_cache`).
3. Read the console: return summary stats, then VaR / CVaR at `alpha` for each selected method (default Normal and Student-t). With `bayes`: predictive VaR / CVaR, posterior mean and HDI, the mu / sigma / nu posterior table with R-hat and ESS, any diagnostic warnings, and the sampling time.
4. Inspect the plot windows (PDF overlay, QQ plots, left-tail density; plus rolling VaR with `--backtest`). Closing them ends the run.
5. With `--backtest`, read the printed summary table (breaches, Kupiec / Christoffersen p-values, reject flags) and the CSVs under `data/backtest/`. With `bayes`, a progress bar per refit and one line with the refit count, cache hits and refits that failed the MCMC diagnostics come first; the per-refit details are in `data/backtest/{coin}-bayes-refits.csv`.
6. If working in git, run `git checkout -- data` to restore the committed data files the run overwrote.

## Data flow

```mermaid
flowchart TD
    U["User: python -m src.main --coin_symbol btc [--backtest]"] --> API
    Y[(Yahoo Finance)] -->|"yfinance, {COIN}-{VS}, period=max, 1d"| API["api.py<br/>fetch_historical_data"]
    API -->|"data/raw/{coin}-{vs}-max.csv<br/>snapped_at, price"| RET["returns.py<br/>compute_log_returns"]
    RET -->|"data/output/{coin}-log-returns.csv<br/>date, return"| DIST["distributions.py<br/>fit_normal / fit_student_t (df >= 2.01)"]
    DIST --> RM["risk_metrics.py<br/>VaR / CVaR: Normal, Student-t, Historical (--methods)"]
    DIST --> VIZ1["visualization.py<br/>PDF overlay, QQ, left-tail density"]
    RM --> OUT1["Console: full-sample VaR / CVaR"]
    RM -->|"--backtest only"| BT["backtest.py<br/>rolling_backtest (window_size)"]
    BT --> BTT["backtest_tests.py<br/>Kupiec POF, Christoffersen ind / cc"]
    BT -->|"data/backtest/{coin}-backtest-results.csv"| F1[(CSV)]
    BTT -->|"data/backtest/{coin}-backtest-summary.csv"| F2[(CSV)]
    BT --> VIZ2["visualization.py<br/>plot_rolling_var"]
    BTT --> VIZ2
    VIZ1 --> SHOW["plt.show()"]
    VIZ2 --> SHOW
    RET -->|"--methods ...,bayes"| BAYES["bayes.py<br/>PyMC Student-t, NUTS (seeded)<br/>VaR / CVaR + HDI, R-hat / ESS"]
    RM -.->|"closed-form t per draw"| BAYES
    BAYES --> OUT1
    BAYES -->|"data/output/{coin}-bayes-summary.csv"| F3[(CSV)]
    BT -->|"bayes in --methods"| RBT["rolling_bayes.py<br/>refit every k days on [r-window, r)<br/>mcmc: Student-t NUTS, warm start<br/>conjugate: NIG closed form"]
    RBT <-->|"mcmc fits, keyed by SHA-256"| CACHE[("data/cache/rolling-bayes/*.npz<br/>(gitignored)")]
    RBT -->|"bayes_var / bayes_cvar"| BT
    RBT -->|"data/backtest/{coin}-bayes-refits.csv"| F4[(CSV)]
```

## CLI paths

| Command | What runs | Files written |
|---|---|---|
| `python -m src.main --coin_symbol btc` | ETL -> fits -> 3 diagnostic plots -> prints Normal and Student-t VaR / CVaR | `data/raw/btc-usd-max.csv`, `data/output/btc-log-returns.csv` |
| `python -m src.main --coin_symbol btc --backtest [--window_size 252] [--alpha 0.05]` | Above, plus rolling backtest (tqdm progress bar, about 2 min for BTC), summary table, rolling VaR plot | Above, plus `data/backtest/btc-backtest-results.csv`, `data/backtest/btc-backtest-summary.csv` |
| `python -m src.api` / `python -m src.returns` | Single ETL step | raw CSV / returns CSV |
| `python -m src.distributions` / `src.risk_metrics` / `src.visualization` | Read the existing returns CSV; print fits / risk metrics incl. Historical (alpha 0.05) / show plots | none |
| `python -m src.main --coin_symbol btc --methods normal,t,bayes --backtest [--bayes_engine mcmc] [--refit_every 21]` | Above, plus the full-sample Bayes and a rolling Bayes backtest column (MCMC every 21 days: about 3-4 min extra for BTC with a cold cache, seconds when cached) | Above, plus `data/output/btc-bayes-summary.csv`, `data/backtest/btc-bayes-refits.csv`, cache files under `data/cache/rolling-bayes/` |
| `python -m src.main ... --backtest --methods normal,t,bayes --bayes_engine conjugate` | Rolling Bayes with the Normal-Inverse-Gamma engine, refit daily (under 1 s extra) | Same files, no cache |
| `python -m src.backtest [--methods normal,student_t,historical,bayes] [--bayes_engine ...] [--save]` | Backtest the existing returns CSV (no download) and print results and runtime | none, or the `data/backtest/` CSVs with `--save` |
| `python -m pytest` | 60 unit tests on synthetic data (no network; Bayes tests use tiny seeded MCMC, about 15 s total; `-m "not slow"` skips the 2 slow tests) | none |
| `python -m src.main --coin_symbol btc --methods normal,t,hist,bayes [--seed 42] [--draws 1000] [--tune 1000] [--chains 4] [--hdi_prob 0.94]` | Default run, plus Historical metrics and the PyMC Bayesian Student-t (about 5 s extra for BTC) | Default-run files, plus `data/output/btc-bayes-summary.csv` |
| `python -m src.bayes --coin_symbol btc` | Bayes only, on the existing returns CSV; prints results | none |

## Backtest timing (no look-ahead)
For each day `i` from `window_size` to `n-1`: fit on returns `[i-window_size, i)`, forecast VaR/CVaR for day `i`, flag a breach if `return_i < VaR_i`. The first `window_size` days have no forecast, so `total_periods = n - window_size`.

Rolling Bayes (`--refit_every k`): refit only on days `r = window_size, window_size + k, ...`, each fit on returns `[r - window_size, r)`, and reuse that posterior for days `r .. r + k - 1`. No forecast ever sees its own day or later; with `k > 1` the posterior can be up to `k - 1` days stale. Details in [TRD.md](TRD.md#rolling-bayesian-backtest-phase-4---backtest---methods-bayes).
