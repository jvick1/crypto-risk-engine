# Crypto Risk Engine

This project develops a modular risk analytics framework for estimating and backtesting Value at Risk (VaR) and Conditional Value at Risk (CVaR) across multiple cryptocurrencies. It implements Normal, Student-t, and Historical simulation approaches. With emphasis on Student-t distribution to capture fat-tailed return behavior commonly digital assets. 

The pipeline is organized to clearly separate raw data, log returns, distribution fitting, rolling window estimation, and out of sample backtesting. The newly added `backtest.py` uses a 252 rolling window to evaluate model performance via breach frequency. The `visualization.py` module provides distribution diagnostics, tail comparisons, and rolling VaR monitoring to support model selection and risk interpretation.

# How to Run
Basic:
```
python -m src.main --coin_symbol btc
```
All Commands:
```
python -m src.main --coin_symbol btc --vs_currency usd --backtest --window_size 252 --alpha 0.05
```

# Folder Structure
```graphql
student_t_var_cvar/
│
├── data/
│   ├── raw/
│   │   └── {coin_symbol}-{vs_currency}-max.csv    # Raw $BTC returns
│   ├── output/
│   │   └── {coin_symbol}-log-returns.csv          # Log Returns
│   └── backtest/
│       └── {coin_symbol}-backtest-results.csv     # Backtest outputs
│
├── src/
│   ├── api.py                                     # Pulls crypto data via yf
│   ├── returns.py                                 # Takes /raw and /output log returns 
│   ├── distributions.py                           # Distribution fitting and risk metrics
│   ├── visualization.py                           # Plots (PDF, QQ, density, and rolling window)
│   ├── risk_metrics.py                            # Calc for Var & Cvar
│   ├── backtest.py                                # Rolling VaR backtest
│   └── main.py                                    # Orchestration Layer
│
└── README.md                                      # Project documentation
```
# Pipeline
## `api.py`
Pull data with yf and saves raw file to `data/raw/{coin_symbol}-{vs_currency}-max.csv`.

## `returns.py`
Responsible for data ingestion and calculating the log returns. Outputs to `data/output/{coin_symbol}-log-returns.csv`.

## `distributions.py`
Fits normal and student-t distributions to log retuns. 

## `visualization.py`
Returns distributions with fitted Normal & Student-t overlay, QQ-plots against each distribution, and left-tail density comparison to highlight extreme risk behavior. Most recent update added rolling window analysis which only populates during a `main.py` call. 

## `risk_metrics.py`
Uses both normal and Student-t the distributions and calculates the Var & Cvar  

## `backtest.py`
Using 3x methods (historical, normal, student-t) backtest VaR w/ rolling window 

## `main.py`
Orchestration Layer - runs ETL (pull & log), distributions, backtesting (if selected), visualization, and cacls risk metrics for `{coin_symbol}`.
