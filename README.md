# Crypto Risk Engine

This project develops a modular risk analytics framework for estimating and backtesting Value at Risk (VaR) and Conditional Value at Risk (CVaR) across multiple cryptocurrencies. It implements Normal, Student-t, and Historical simulation approaches (parametric Normal/Student-t and historical VaR/CVaR are all in `risk_metrics.py`), plus an optional Bayesian Student-t (PyMC, `bayes.py`) that adds credible intervals to VaR/CVaR, and an optional rolling Bayesian backtest (`rolling_bayes.py`). With emphasis on Student-t distribution to capture fat-tailed return behavior commonly seen in digital assets. 

The pipeline is organized to clearly separate raw data, log returns, distribution fitting, rolling window estimation, and out of sample backtesting. The `backtest.py` module uses a rolling window (default 252 observations) to evaluate model performance via breach frequency. The `visualization.py` module provides distribution diagnostics, tail comparisons, and rolling VaR monitoring to support model selection and risk interpretation.

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
Every run downloads fresh data from Yahoo Finance, so an internet connection is required.

Basic:
```
python -m src.main --coin_symbol btc
```
All Commands:
```
python -m src.main --coin_symbol btc --vs_currency usd --backtest --window_size 252 --alpha 0.05
```
Bayesian Student-t (opt-in, adds about 5 s for BTC):
```
python -m src.main --coin_symbol btc --vs_currency usd --alpha 0.05 --methods normal,t,hist,bayes --seed 42
```
Rolling Bayesian backtest (adds a `bayes` column to the backtest; MCMC refit every 21 days is about 3-4 extra minutes for BTC the first time, much faster when the fits are cached):
```
python -m src.main --coin_symbol btc --backtest --methods normal,t,bayes
python -m src.main --coin_symbol btc --backtest --methods normal,t,bayes --bayes_engine conjugate   # closed form, refit daily, under 1 s
```
Flags:
- `--coin_symbol` (default `btc`) and `--vs_currency` (default `usd`) build the Yahoo ticker, e.g. `BTC-USD`.
- `--alpha` is a **tail probability** (default `0.05`, i.e. 95% VaR), not a confidence level.
- `--backtest` is required to run the rolling backtest and write `data/backtest/{coin_symbol}-backtest-results.csv` plus `data/backtest/{coin_symbol}-backtest-summary.csv` (per-method breaches, expected breaches, breach rate, Kupiec POF / Christoffersen independence / conditional coverage LR stats, p-values and reject-at-5% flags); without it only the full-sample fit and risk metrics are produced.
- `--window_size` is the rolling window length in observations (default `252`; crypto trades every day, so this is about 8 months).
- `--methods` picks the full-sample methods to print, comma-separated from `normal`, `t`, `hist`, `bayes` (default `normal,t`, the original output). `bayes` runs MCMC, so it is off by default. `--backtest` always runs Normal, Student-t and Historical; listing `bayes` also adds the rolling Bayes backtest.
- Bayes only: `--seed` (default `42`; same seed and data give the same result), `--draws` (default `1000` per chain), `--tune` (default `1000` warm-up steps per chain), `--chains` (default `4`, run in parallel) and `--hdi_prob` (default `0.94`). With `bayes`, the run also writes `data/output/{coin_symbol}-bayes-summary.csv`.
- Rolling Bayes backtest only (with `--backtest` and `bayes` in `--methods`):
  - `--bayes_engine`: `mcmc` (default; the Bayesian Student-t above, refit with PyMC) or `conjugate` (Normal likelihood with a Normal-Inverse-Gamma prior; closed form, very fast). The conjugate engine's tails come only from parameter uncertainty, so with 252 days it is close to the Normal VaR; it is a fast baseline, not a fat-tail model.
  - `--refit_every` (default `21` for mcmc, `1` for conjugate): refit every k days and reuse the last posterior in between. Each fit only uses data from before its refit day, so there is no look-ahead, but the posterior can be up to k - 1 days stale.
  - `--bt_draws` (default `500`), `--bt_tune` (default `500`) and `--bt_chains` (default `2`): MCMC settings per refit, smaller than the full-sample run. Each refit starts from the previous refit's posterior means (warm start). `--seed` also seeds the refits.
  - `--no_cache`: MCMC fits are cached in `data/cache/rolling-bayes/` (gitignored), keyed by the window data and settings, so reruns are fast; this flag turns the cache off.
  - Writes `bayes_var` / `bayes_cvar` / `bayes_breach` to the results CSV, a `bayes` row to the summary CSV and a per-refit log (diagnostics, R-hat / ESS / divergences, cache hits) to `data/backtest/{coin_symbol}-bayes-refits.csv`. Refits that fail the diagnostics are counted in one console line instead of one warning each.

Tests:
```
python -m pytest                 # all 60 tests (about 15 s)
python -m pytest -m "not slow"   # skip the 2 slower MCMC tests
```

# Folder Structure
```graphql
crypto-risk-engine/
│
├── data/
│   ├── raw/
│   │   └── {coin_symbol}-{vs_currency}-max.csv    # Raw daily prices (any coin)
│   ├── output/
│   │   ├── {coin_symbol}-log-returns.csv          # Log Returns
│   │   └── {coin_symbol}-bayes-summary.csv        # Bayesian Student-t summary (--methods bayes)
│   └── backtest/
│       ├── {coin_symbol}-backtest-results.csv     # Backtest outputs
│       ├── {coin_symbol}-backtest-summary.csv     # Breach tests summary
│       └── {coin_symbol}-bayes-refits.csv         # Rolling Bayes refit log (with bayes)
│   (data/cache/ holds cached rolling Bayes fits; gitignored)
│
├── src/
│   ├── __init__.py                                # Package marker
│   ├── api.py                                     # Pulls crypto data via yf
│   ├── returns.py                                 # Takes /raw and /output log returns 
│   ├── distributions.py                           # Normal & Student-t distribution fitting
│   ├── visualization.py                           # Plots (PDF, QQ, density, and rolling window)
│   ├── risk_metrics.py                            # VaR & CVaR (Normal, Student-t, Historical)
│   ├── bayes.py                                   # Bayesian Student-t (PyMC): VaR/CVaR + credible intervals
│   ├── backtest.py                                # Rolling VaR backtest
│   ├── rolling_bayes.py                           # Rolling Bayesian VaR (MCMC / conjugate), refit schedule, cache
│   ├── backtest_tests.py                          # Kupiec / Christoffersen tests
│   └── main.py                                    # Orchestration Layer
│
├── docs/                                          # PRD, TRD, app flow, schema, roadmap
├── tests/                                         # pytest sanity tests
├── pytest.ini                                     # registers the `slow` test marker
├── .gitignore
├── requirements.txt                               # Minimum dependency versions
└── README.md                                      # Project documentation
```
# Pipeline
## `api.py`
Pull data with yf and saves raw file to `data/raw/{coin_symbol}-{vs_currency}-max.csv`.

## `returns.py`
Responsible for data ingestion and calculating the log returns. Outputs to `data/output/{coin_symbol}-log-returns.csv`.

## `distributions.py`
Fits normal and student-t distributions to log returns. 

## `visualization.py`
Returns distributions with fitted Normal & Student-t overlay, QQ-plots against each distribution, and left-tail density comparison to highlight extreme risk behavior. Most recent update added rolling window analysis which is only produced by a `main.py` call with `--backtest`. 

## `risk_metrics.py`
Calculates VaR & CVaR from the fitted Normal and Student-t distributions, plus Historical (empirical) VaR & CVaR. VaR/CVaR are quantiles of log returns and are negative numbers for losses (left tail).

## `bayes.py`
Bayesian Student-t on the full sample with PyMC (`--methods bayes`). Weakly informative priors scaled to the data: location ~ Normal(0, half a std), scale ~ HalfNormal(1 std), and degrees of freedom nu = 2 + Gamma(2, 0.1), so nu is always above 2 and the variance stays finite. It reports:
- the posterior predictive VaR/CVaR (one number that accounts for parameter uncertainty),
- the posterior mean and a 94% highest-density credible interval for VaR and CVaR, computed from the closed-form Student-t formulas for each posterior draw,
- posterior summaries for location, scale and nu, with R-hat and bulk/tail effective sample size. It warns if any R-hat is above 1.01, any ESS is below 400, or the sampler has divergences.

Results are saved to `data/output/{coin_symbol}-bayes-summary.csv` (columns in [docs/SCHEMA.md](docs/SCHEMA.md)). Run it alone on the existing returns CSV with `python -m src.bayes --coin_symbol btc`.

## `backtest.py`
Using 3x methods (historical, normal, student-t) backtest VaR w/ rolling window, then run Kupiec POF and Christoffersen independence / conditional coverage tests (`backtest_tests.py`). Rolling windows overlap, so the tests' independence assumption is approximate. 

## `rolling_bayes.py`
Adds a 4th backtest method, `bayes`, when `--methods` includes `bayes`. Refits happen every `--refit_every` days on the `window_size` returns before the refit day, and the posterior is reused until the next refit (no look-ahead). Two engines: `mcmc` (the `bayes.py` Student-t, compiled once, warm-started from the previous posterior means, with R-hat / ESS / divergence checks per refit and cached fits) and `conjugate` (Normal-Inverse-Gamma, closed-form Student-t predictive, refit daily). Its results get the same Kupiec / Christoffersen tests as the other methods. Method details and caveats are in [docs/TRD.md](docs/TRD.md).

## `main.py`
Orchestration Layer - runs ETL (pull & log), distributions, backtesting (only with `--backtest`), visualization, and calcs risk metrics for `{coin_symbol}` for the selected `--methods` (Bayesian Student-t only with `bayes`).

# Plots

## Probability Density Functions
This chart shows empirical distribution of log returns as a histogram overlaid with Normal and Student-t probability density functions (PDFs). This is useful in VaR & CVaR analysis for visually assessing how well the models capture the fat tails and skewness of the data.

<img width="500" alt="Prob Density Func" src="https://github.com/user-attachments/assets/734b9654-e22a-4aeb-9c14-d419c1d9c80b" />

## Quantile-Quantile Plot
QQ plots compares the quantiles of the log returns against Normal & Student-t distributions. Highlights how Student-t better accommodates heavy tailed behavior in digital assets.

<img height="300" alt="QQ plot" src="https://github.com/user-attachments/assets/f33fa17b-ff46-428b-abb2-ee6e135ad2b8" />

## Left-tail Density
This plot compares the left-tail densities of fitted Normal and Student-t distributions for returns. Typically, Student-t tails decay slower helping to accurately predict extreme downside risk.

<img height="300" alt="Image" src="https://github.com/user-attachments/assets/3d9a0a70-c576-44c8-9a17-3632725620f4" />

## Backtest
This backtest chart shows the rolling VaR (`--alpha`, default 5%) est for returns using Normal, Student-t, and Historical methods overlaid on actual returns. This helps compare model performance over time, spot periods of risk, and validate models against historical crashes. 

<img height="400" alt="Image" src="https://github.com/user-attachments/assets/35557aad-8fc7-410e-8c03-490eb3535509" />



See [docs/ROADMAP.md](docs/ROADMAP.md) for planned phases. Educational project, not investment advice.
