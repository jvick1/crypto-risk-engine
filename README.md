# Crypto Risk Engine

This project is a modular risk analytics framework for estimating and backtesting Value at Risk (VaR) and Conditional Value at Risk (CVaR) on cryptocurrency daily log returns. It implements Normal, Student-t, and Historical simulation methods, an optional Bayesian Student-t (PyMC) that adds credible intervals, and an optional rolling Bayesian backtest.

**Educational project, not investment advice.**

The pipeline separates raw prices, log returns, distribution fitting, risk metrics, rolling-window estimation, and out-of-sample backtesting. `backtest.py` uses a rolling window (default 252 observations) and judges each method with Kupiec POF and Christoffersen independence / conditional coverage tests. `visualization.py` provides distribution diagnostics, tail comparisons, and rolling VaR monitoring.

# Docs
- [Product requirements (PRD)](docs/PRD.md)
- [Technical requirements (TRD)](docs/TRD.md)
- [App flow](docs/APP_FLOW.md)
- [Data schema](docs/SCHEMA.md)
- [Roadmap / implementation plan](docs/ROADMAP.md)
- [Design brief](docs/DESIGN_BRIEF.md): brand, look and voice (draft)

# Install
Needs Python 3.11 or newer.

**Anaconda (recommended on Windows):** use a separate env, not base. Installing PyMC into base can upgrade NumPy past what Anaconda's own packages were built for.
```
conda env create -f environment.yml
conda activate crypto-risk
```

**pip:**
```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

# How to Run
Every `python -m src.main` run downloads fresh data from Yahoo Finance (an internet connection is required) and overwrites the committed CSVs under `data/`. Restore them before switching branches or committing:

```
git restore data/
```

## Example commands

Basic full-sample fit (Normal and Student-t VaR/CVaR, three diagnostic plots):
```
python -m src.main --coin_symbol btc
```

Rolling backtest (always runs Normal, Student-t and Historical):
```
python -m src.main --coin_symbol btc --vs_currency usd --backtest --window_size 252 --alpha 0.05
```
About 92 s for BTC (Student-t MLE loop), measured on the committed BTC run in [docs/ROADMAP.md](docs/ROADMAP.md).

Bayes full-sample (adds Historical printout and the PyMC Student-t):
```
python -m src.main --coin_symbol btc --vs_currency usd --alpha 0.05 --methods normal,t,hist,bayes --seed 42
```
Sampling about 5 s, full run about 7 s, measured on the BTC run of 2026-11-12 in the roadmap.

Rolling Bayes backtest, conjugate engine (closed form, refit daily):
```
python -m src.main --coin_symbol btc --backtest --methods normal,t,bayes --bayes_engine conjugate
```
Bayes part 0.7 s on the same BTC run.

Rolling Bayes backtest, MCMC engine (default; refit every 21 days):
```
python -m src.main --coin_symbol btc --backtest --methods normal,t,bayes
```
Bayes part 178 s with a cold cache, 0.6 s when fully cached; whole backtest about 270 s, measured on the same BTC run.

You can also run individual modules against an existing CSV, with no download: `python -m src.api`, `src.returns`, `src.distributions`, `src.risk_metrics`, `src.visualization`, `src.backtest` (add `--save` to write backtest CSVs), and `python -m src.bayes --coin_symbol btc` (prints only, does not write the summary CSV). See [docs/APP_FLOW.md](docs/APP_FLOW.md) and [docs/TRD.md](docs/TRD.md#interfaces).

## Flags (`python -m src.main`)

These are all of the flags from `python -m src.main --help`. There is no `--verbose` / `--no-verbose` on `src.main` (it always prints); those flags exist only on `python -m src.returns`.

| Flag | Default | Meaning |
|---|---|---|
| `-h`, `--help` | — | Show the help message and exit |
| `--coin_symbol` | `btc` | Coin symbol (e.g. `btc`, `eth`, `sol`). Upper-cased into the Yahoo ticker and used in file names |
| `--vs_currency` | `usd` | Quote currency. The ticker is `{COIN}-{VS}`, e.g. `BTC-USD` |
| `--alpha` | `0.05` | Tail probability for VaR/CVaR (`0.05` means 95% VaR), not a confidence level |
| `--backtest` | off | Run the out-of-sample rolling backtest, write the backtest CSVs, and show the rolling VaR plot |
| `--window_size` | `252` | Rolling window length in observations (crypto trades every day, so this is about 8 months) |
| `--methods` | `normal,t` | Comma-separated full-sample methods to print: `normal`, `t`, `hist`, `bayes`. `bayes` runs MCMC, so it is off by default. `--backtest` always runs Normal, Student-t and Historical, and adds the rolling Bayes backtest when `bayes` is listed |
| `--seed` | `42` | Random seed for the Bayesian sampler and posterior predictive. Same seed and data give the same result. Also seeds rolling MCMC refits (`seed +` the refit day) |
| `--draws` | `1000` | Bayes (full sample): posterior draws per chain |
| `--tune` | `1000` | Bayes (full sample): tuning (warm-up) steps per chain, discarded |
| `--chains` | `4` | Bayes (full sample): MCMC chains, run in parallel up to the CPU count |
| `--hdi_prob` | `0.94` | Bayes (full sample): highest-density credible interval mass |
| `--bayes_engine` | `mcmc` | Backtest Bayes engine: `mcmc` (Student-t, NUTS) or `conjugate` (Normal-Inverse-Gamma, closed form, fast). Only used with `--backtest` and `bayes` in `--methods` |
| `--refit_every` | `21` for `mcmc`, `1` for `conjugate` | Backtest Bayes: refit every k days, reusing the last posterior in between |
| `--bt_draws` | `500` | Backtest Bayes (`mcmc`): draws per chain per refit |
| `--bt_tune` | `500` | Backtest Bayes (`mcmc`): tuning steps per chain per refit |
| `--bt_chains` | `2` | Backtest Bayes (`mcmc`): chains per refit, run sequentially |
| `--no_cache` | off | Backtest Bayes (`mcmc`): do not read or write cached fits in `data/cache/rolling-bayes/` |
| `--log-level` | `WARNING` | Python logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). `INFO` shows per-stage start/end timing lines |
| `--profile` | off | Print a stage timing table at the end of the run and write `data/output/{coin_symbol}-timings.csv` (gitignored) |
| `--no_plots` | off | Skip plot generation and `plt.show()` (for timing runs and non-interactive use) |

Profile a run (example, conjugate backtest, no plots):
```
python -m src.main --coin_symbol btc --backtest --methods normal,t,hist,bayes --bayes_engine conjugate --profile --no_plots
```
Measured stage times and a ranked bottleneck table are in [docs/APP_FLOW.md](docs/APP_FLOW.md).

## Methods

- **Normal.** MLE Normal on log returns; VaR/CVaR are closed-form left-tail quantiles (negative = loss). In the rolling backtest, Normal uses the window mean and std (pandas `ddof=1`); the full-sample fit uses `ddof=0`.
- **Student-t.** MLE Student-t (`scipy.stats.t.fit`) with a degrees-of-freedom floor of 2.01 (variance is finite only for df > 2). If the unconstrained MLE is below the floor, the fit is redone with df fixed at 2.01. If the fit fails or returns invalid parameters, it falls back to a Normal-equivalent t (df = 1e6) and warns. CVaR is guarded so it never returns NaN/inf for valid inputs.
- **Historical.** Empirical VaR (percentile of the sample or window) and CVaR (mean of returns at or below that VaR). Printed in the full-sample output only when `hist` is in `--methods`; always included in `--backtest`.
- **Bayes, full sample** (`--methods bayes`, `src/bayes.py`). PyMC Student-t sampled with NUTS. Weakly informative priors, scaled to the sample std `s`: location `mu ~ Normal(0, 0.5 s)`, scale `sigma ~ HalfNormal(s)`, and `nu = 2 + Gamma(2, 0.1)` so nu is always above 2. It reports (1) the exact posterior-predictive VaR/CVaR from the mixture of per-draw t's, (2) the posterior mean, sd, median and a 94% highest-density credible interval for VaR and CVaR from the closed-form Student-t formulas on each draw, and (3) posterior summaries for location, scale and nu, with R-hat and bulk/tail ESS. It warns if any R-hat is above 1.01, any ESS is below 400, or the sampler has divergences. Results go to `data/output/{coin_symbol}-bayes-summary.csv`.
- **Rolling Bayes backtest** (`--backtest` with `bayes` in `--methods`, `src/rolling_bayes.py`). Refits every `--refit_every` k days on the `window_size` returns ending the day before the refit day, and reuses that posterior until the next refit (no look-ahead; with k > 1 the posterior can be up to k − 1 days stale). Two engines:
  - `mcmc` (default): the same Bayesian Student-t as above, compiled once, warm-started from the previous refit's posterior means (priors stay fixed; warm start only helps the sampler). Per-refit R-hat / ESS / divergence checks; failures are counted in one console line. Fits are cached under `data/cache/rolling-bayes/` (gitignored), keyed by the window data, settings, seed and warm-start values, unless `--no_cache`.
  - `conjugate`: Normal likelihood with a Normal-Inverse-Gamma prior; the posterior predictive is a closed-form Student-t, fast enough to refit daily. Fat tails come only from parameter uncertainty, so with a 252-day window it is essentially Normal (on the committed BTC run it matched Normal exactly: same 199 breach days). It is a fast baseline, not a fat-tail model, and is not cached.

## Outputs

VaR/CVaR in every file are log-return quantiles at `--alpha` (negative = loss). Plot functions return matplotlib Figures; `src.main` shows them with `plt.show()` unless `--no_plots` is set. Plots are **not saved to disk** (the README images are hosted on GitHub). Column layouts are in [docs/SCHEMA.md](docs/SCHEMA.md).

| Path | Written when | Committed in the repo? |
|---|---|---|
| `data/raw/{coin_symbol}-{vs_currency}-max.csv` | Every `src.main` run (and `python -m src.api`) | Yes, examples for `btc`, `eth`, `sol`, `doge`, `xrp` vs `usd`. A run overwrites the matching file |
| `data/output/{coin_symbol}-log-returns.csv` | Every `src.main` run (and `python -m src.returns`) | Yes, examples for the same five coins. A run overwrites the matching file |
| `data/output/{coin_symbol}-bayes-summary.csv` | `--methods` includes `bayes` | Yes, example: `btc` (from the 2026-11-12 live run; newer than the committed returns CSV) |
| `data/backtest/{coin_symbol}-backtest-results.csv` | `--backtest` | Yes, examples: `btc` (Phase 4 MCMC run on the committed returns) and `sol` (earlier run, before the df floor, no Bayes columns). A run overwrites the matching file |
| `data/backtest/{coin_symbol}-backtest-summary.csv` | `--backtest` | No (generated only) |
| `data/backtest/{coin_symbol}-bayes-refits.csv` | `--backtest` with `bayes` in `--methods` | No (generated only) |
| `data/cache/rolling-bayes/{sha256}.npz` | Rolling MCMC refits, unless `--no_cache` | No; the `data/cache/` directory is gitignored |
| `data/output/{coin_symbol}-timings.csv` | `--profile` | No; gitignored (`data/output/*-timings.csv`). One row per stage: `run_id`, `stage`, `parent`, `seconds`, plus the run settings |

`--backtest` always writes `bayes_var` / `bayes_cvar` / `bayes_breach` into the results CSV, a `bayes` row into the summary CSV, and the refit log, when `bayes` is listed. Without `bayes`, the first 11 results columns stay the same.

## Backtest tests

The summary CSV and console table report, at a 5% significance level:

- **Kupiec POF** (unconditional coverage; LR ~ chi-squared with 1 df). Null: the breach probability equals `alpha`. Rejects if the model is breached too often or too rarely.
- **Christoffersen independence** (LR ~ chi-squared with 1 df). Null: breaches are independent (no clustering).
- **Christoffersen conditional coverage** (LR = POF + independence, chi-squared with 2 df). Both coverage and independence.

Reject flags (`reject_pof_5pct`, `reject_ind_5pct`, `reject_cc_5pct`) are true when the p-value is below 0.05. Rolling windows overlap, so the tests' independence assumption is approximate; treat p-values as approximate. Edge cases (zero breaches, no transitions) return finite values (`0 * log(0) = 0`).

## Tests
```
python -m pytest                 # all 67 tests (about 15 s)
python -m pytest -m "not slow"   # skip the 2 slower MCMC tests
```

## Caveats

- Educational only, not investment advice.
- `--alpha` is a tail probability (default 0.05, i.e. 95% VaR), not a confidence level. `--window_size` is a count of daily observations (default 252, about 8 months of crypto trading).
- Rolling windows overlap, so Kupiec and Christoffersen p-values are approximate.
- On the committed BTC backtest (returns 2014-09-18 to 2026-02-28, 3,930 forecast days 2015-05-28 to 2026-02-28, alpha 0.05, window 252, expected breaches 196.5; figures from [docs/ROADMAP.md](docs/ROADMAP.md)):

  | Method | Breaches | Kupiec p | Independence p |
  |---|---|---|---|
  | Normal | 199 (5.06%) | 0.855 | 0.0013 |
  | Student-t | 234 (5.95%) | 0.0076 | 0.0003 |
  | Historical | 214 (5.45%) | 0.206 | 0.035 |
  | Bayes conjugate (k=1) | 199 (5.06%), same days as Normal | 0.855 | 0.0013 |
  | Bayes MCMC (k=21, seed 42) | 248 (6.31%) | 0.0003 | 0.0005 |

  All five methods fail the Christoffersen independence test because volatility clusters; GARCH is in the backlog. Bayes MCMC at k=21 also fails Kupiec on this BTC run (10 of 188 refits had R-hat between 1.01 and 1.02). The conjugate engine equals Normal here (same breach days; VaR within 0.00004), as expected from a Normal likelihood.
- Every `src.main` run re-downloads data and rewrites the committed CSVs under `data/`. Use `git restore data/` before switching branches or committing, or the checkout will refuse to proceed.

# Folder Structure
```
crypto-risk-engine/
├── data/
│   ├── raw/
│   │   └── {coin_symbol}-{vs_currency}-max.csv    # Raw daily prices
│   ├── output/
│   │   ├── {coin_symbol}-log-returns.csv          # Log returns
│   │   └── {coin_symbol}-bayes-summary.csv        # Bayesian Student-t summary (--methods bayes)
│   ├── backtest/
│   │   ├── {coin_symbol}-backtest-results.csv     # Per-day VaR, CVaR, breaches
│   │   ├── {coin_symbol}-backtest-summary.csv     # Kupiec / Christoffersen summary (not committed)
│   │   └── {coin_symbol}-bayes-refits.csv         # Rolling Bayes refit log (not committed)
│   └── cache/rolling-bayes/                       # Cached MCMC fits (gitignored)
│
├── src/
│   ├── __init__.py                                # Package marker
│   ├── api.py                                     # Pulls crypto data via yfinance
│   ├── returns.py                                 # Raw prices -> log returns
│   ├── distributions.py                           # Normal & Student-t distribution fitting
│   ├── visualization.py                           # Plots (PDF, QQ, left-tail, rolling VaR)
│   ├── risk_metrics.py                            # VaR & CVaR (Normal, Student-t, Historical)
│   ├── bayes.py                                   # Bayesian Student-t (PyMC): VaR/CVaR + credible intervals
│   ├── backtest.py                                # Rolling VaR backtest
│   ├── rolling_bayes.py                           # Rolling Bayesian VaR (MCMC / conjugate), refit schedule, cache
│   ├── backtest_tests.py                          # Kupiec / Christoffersen tests
│   ├── timing.py                                  # Nestable stage timer, --profile CSV, on_stage hook
│   └── main.py                                    # Orchestration layer
│
├── docs/                                          # PRD, TRD, app flow, schema, roadmap, design brief
├── tests/                                         # pytest (67 tests; 2 marked slow)
├── pytest.ini                                     # Registers the `slow` test marker
├── environment.yml                                # Conda env (Python 3.11, conda-forge PyMC)
├── requirements.txt                               # Minimum pip dependency versions
├── .gitignore
└── README.md
```

# Pipeline
## `api.py`
Pulls daily closes from Yahoo Finance (`period=max`, `interval=1d`, `auto_adjust=False`) and saves `data/raw/{coin_symbol}-{vs_currency}-max.csv`.

## `returns.py`
Ingests the raw file, sorts and de-duplicates by date, and writes daily log returns to `data/output/{coin_symbol}-log-returns.csv`.

## `distributions.py`
Fits Normal and Student-t distributions to log returns, with the df floor and Normal fallback described above.

## `visualization.py`
Return histogram with fitted Normal and Student-t PDFs, QQ plots against each distribution, and a left-tail density comparison. The rolling VaR plot is produced only by a `main.py` (or equivalent) call with `--backtest`.

## `risk_metrics.py`
Calculates VaR and CVaR from the fitted Normal and Student-t distributions, plus Historical (empirical) VaR and CVaR. Values are left-tail quantiles of log returns and are negative for losses.

## `bayes.py`
Bayesian Student-t on the full sample with PyMC (`--methods bayes`), as described under Methods. Run it alone on an existing returns CSV with `python -m src.bayes --coin_symbol btc` (prints only; `main.py` is what writes the summary CSV).

## `backtest.py`
Rolling-window VaR backtest for Historical, Normal and Student-t, then Kupiec POF and Christoffersen independence / conditional coverage tests (`backtest_tests.py`). Each forecast for day `i` is fit on returns `[i - window_size, i)` (no look-ahead).

## `rolling_bayes.py`
Adds a fourth backtest method, `bayes`, when `--methods` includes `bayes`. See Methods for the engines, refit schedule, warm start and cache. Its results get the same Kupiec / Christoffersen tests as the other methods. Further detail is in [docs/TRD.md](docs/TRD.md).

## `main.py`
Orchestration layer: ETL (download and log returns), distribution fits, visualization, risk metrics for the selected `--methods` (Bayesian Student-t only with `bayes`), and the rolling backtest only with `--backtest`.

# Plots

## Probability Density Functions
This chart shows the empirical distribution of log returns as a histogram overlaid with Normal and Student-t probability density functions (PDFs). Useful in VaR and CVaR analysis for visually assessing how well the models capture the fat tails and skewness of the data.

<img width="500" alt="Prob Density Func" src="https://github.com/user-attachments/assets/734b9654-e22a-4aeb-9c14-d419c1d9c80b" />

## Quantile-Quantile Plot
QQ plots compare the quantiles of the log returns against Normal and Student-t distributions. Highlights how Student-t better accommodates heavy-tailed behavior in digital assets.

<img height="300" alt="QQ plot" src="https://github.com/user-attachments/assets/f33fa17b-ff46-428b-abb2-ee6e135ad2b8" />

## Left-tail Density
This plot compares the left-tail densities of fitted Normal and Student-t distributions for returns. Typically, Student-t tails decay more slowly, which helps capture extreme downside risk.

<img height="300" alt="Image" src="https://github.com/user-attachments/assets/3d9a0a70-c576-44c8-9a17-3632725620f4" />

## Backtest
This backtest chart shows the rolling VaR (`--alpha`, default 5%) for returns using Normal, Student-t, and Historical methods (plus Bayes when `bayes` is in `--methods`) overlaid on actual returns. It helps compare model performance over time, spot periods of risk, and validate models against historical crashes.

<img height="400" alt="Image" src="https://github.com/user-attachments/assets/35557aad-8fc7-410e-8c03-490eb3535509" />

See [docs/ROADMAP.md](docs/ROADMAP.md) for planned phases.
