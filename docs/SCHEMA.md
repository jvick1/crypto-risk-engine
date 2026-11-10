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
Written by `main.py` with `--backtest`. One row per out-of-sample day (`n - window_size` rows). `{m}` is each of `normal`, `student_t`, `historical`.

| Column | Type | Meaning |
|---|---|---|
| `date` | date | Forecast day |
| `actual_return` | float | Realized log return that day |
| `{m}_var` | float (negative = loss) | VaR at `alpha` from the prior `window_size` returns |
| `{m}_cvar` | float (negative = loss) | CVaR / expected shortfall at `alpha` |
| `{m}_breach` | bool | `actual_return < {m}_var` |

Full column order: `date, actual_return, normal_var, normal_cvar, normal_breach, student_t_var, student_t_cvar, student_t_breach, historical_var, historical_cvar, historical_breach`. Committed examples: `btc`, `sol` (from an earlier run, before the Phase 2 df floor).

### 4. Backtest summary: `data/backtest/{coin}-backtest-summary.csv`
Written by `main.py` with `--backtest` (added in Phase 1; none committed yet). One row per method.

| Column | Type | Meaning |
|---|---|---|
| `method` | str (index) | `normal`, `student_t`, `historical` |
| `breaches` | int | Count of breach days |
| `total_periods` | int | Out-of-sample days |
| `expected_breaches` | float | `alpha * total_periods` |
| `breach_rate` | float | `breaches / total_periods` |
| `expected_rate` | float | `alpha` |
| `lr_pof`, `p_pof` | float | Kupiec POF LR statistic and p-value (chi2, 1 df) |
| `lr_ind`, `p_ind` | float | Christoffersen independence LR and p-value (chi2, 1 df) |
| `lr_cc`, `p_cc` | float | Conditional coverage LR (`lr_pof + lr_ind`) and p-value (chi2, 2 df) |
| `reject_pof_5pct`, `reject_ind_5pct`, `reject_cc_5pct` | bool | p-value < 0.05 |

## Plot outputs
Figures are shown interactively with `plt.show()` at the end of `main`; **they are not saved to disk** (the README images are hosted on GitHub).

| Figure | Function | When |
|---|---|---|
| Return histogram with Normal and Student-t PDFs and VaR lines | `plot_histogram_with_fits` | Every run |
| QQ plots vs Normal and vs Student-t | `plot_qq_plots` | Every run |
| Left-tail density, log scale (x from -0.30 to 0) | `plot_tail_comparison` | Every run |
| Rolling VaR per method with breach markers and breach rate in legend | `plot_rolling_var` | `--backtest` only |

## Config
There is no config file or environment variable. Configuration is the CLI flags (see [TRD.md](TRD.md#interfaces)) plus these code constants:

| Constant | Value | Where |
|---|---|---|
| `DF_FLOOR` | `2.01` | `src/distributions.py` (minimum Student-t df) |
| `DF_NORMAL_FALLBACK` | `1e6` | `src/distributions.py` (df used when the t fit fails) |
| Test significance `sig` | `0.05` | `src/backtest_tests.christoffersen_cc` |
| Yahoo download | `period="max"`, `interval="1d"`, `auto_adjust=False` | `src/api.py` |
| Default methods | `["normal", "student_t", "historical"]` | `src/backtest.rolling_backtest` |

**Planned** (Phase 3-4): Bayesian outputs (posterior VaR/CVaR with credible intervals, R-hat/ESS) and a fit cache; schema to be added here when built.
