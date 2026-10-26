# Crypto Risk Engine

This project develops a modular risk analytics framework for estimating and backtesting Value at Risk (VaR) and Conditional Value at Risk (CVaR) across multiple cryptocurrencies. It implements Normal, Student-t, and Historical simulation approaches (parametric Normal/Student-t and historical VaR/CVaR are all in `risk_metrics.py`). With emphasis on Student-t distribution to capture fat-tailed return behavior commonly seen in digital assets. 

The pipeline is organized to clearly separate raw data, log returns, distribution fitting, rolling window estimation, and out of sample backtesting. The `backtest.py` module uses a rolling window (default 252 observations) to evaluate model performance via breach frequency. The `visualization.py` module provides distribution diagnostics, tail comparisons, and rolling VaR monitoring to support model selection and risk interpretation.

# Install
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
Flags:
- `--coin_symbol` (default `btc`) and `--vs_currency` (default `usd`) build the Yahoo ticker, e.g. `BTC-USD`.
- `--alpha` is a **tail probability** (default `0.05`, i.e. 95% VaR), not a confidence level.
- `--backtest` is required to run the rolling backtest and write `data/backtest/{coin_symbol}-backtest-results.csv`; without it only the full-sample fit and risk metrics are produced.
- `--window_size` is the rolling window length in observations (default `252`; crypto trades every day, so this is about 8 months).

Tests:
```
python -m pytest
```

# Folder Structure
```graphql
crypto-risk-engine/
│
├── data/
│   ├── raw/
│   │   └── {coin_symbol}-{vs_currency}-max.csv    # Raw daily prices (any coin)
│   ├── output/
│   │   └── {coin_symbol}-log-returns.csv          # Log Returns
│   └── backtest/
│       └── {coin_symbol}-backtest-results.csv     # Backtest outputs
│
├── src/
│   ├── __init__.py                                # Package marker
│   ├── api.py                                     # Pulls crypto data via yf
│   ├── returns.py                                 # Takes /raw and /output log returns 
│   ├── distributions.py                           # Normal & Student-t distribution fitting
│   ├── visualization.py                           # Plots (PDF, QQ, density, and rolling window)
│   ├── risk_metrics.py                            # VaR & CVaR (Normal, Student-t, Historical)
│   ├── backtest.py                                # Rolling VaR backtest
│   └── main.py                                    # Orchestration Layer
│
├── tests/                                         # pytest sanity tests
├── .gitignore
├── requirements.txt                               # Pinned dependencies
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

## `backtest.py`
Using 3x methods (historical, normal, student-t) backtest VaR w/ rolling window 

## `main.py`
Orchestration Layer - runs ETL (pull & log), distributions, backtesting (only with `--backtest`), visualization, and calcs risk metrics for `{coin_symbol}`.

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

