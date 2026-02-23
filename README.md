# Crypto Risk Engine

This project implements Value at Risk (VaR) and Conditional Value at Risk (CVaR) using the Student’s t-distribution, capturing fat-tailed behavior commonly observed in financial returns on multiple Cryptocurrencies. The pipeline is organized to clearly separate raw data, log returns, distribution fitting, and visualization for reproducible risk modeling.

# How to Run
```
python -m src.main --coin_symbol eth
```

# Folder Structure
```graphql
student_t_var_cvar/
│
├── data/
│   ├── raw/
│   │   └── {coin_symbol}-{vs_currency}-max.csv        # Raw $BTC returns
│   │
│   └── output/
│       └── {coin_symbol}-log-returns.csv              # Log Returns 
│
├── src/
│   ├── api.py                     # Pulls crypto data via yf
│   ├── returns.py                 # Takes /raw and /output log returns 
│   ├── distributions.py           # Distribution fitting and risk metrics
│   ├── visualization.py           # Plots (PDF, QQ, and density)
│   ├── risk_metrics.py            # Calc for Var & Cvar
│   └── main.py                    # Orchestration Layer
│
└── README.md                      # Project documentation
```
# Pipeline
## `api.py`
Pull data with yf save to raw.

## `returns.py`
Responsible for data ingestion and outputs log returns.

## `distributions.py`
Fits normal and student-t distributions to log retuns. 

## `visualization.py`
Returns distributions with fitted Normal & Student-t overlay, QQ-plots against each distribution, and left-tail density comparison to highlight extreme risk behavior. 

## `risk_metrics.py`
Uses both normal and Student-t the distributions and calculates the Var & Cvar  

## `main.py`
Orchestration Layer - runs ets, dist, viz and cacls risk metrics for static file.
