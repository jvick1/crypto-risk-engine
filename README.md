# Student-t VaR / CVaR modeling

This project implements Value at Risk (VaR) and Conditional Value at Risk (CVaR) using the Student’s t-distribution, capturing fat-tailed behavior commonly observed in financial returns ($BTC). The pipeline is organized to clearly separate raw data, log returns, distribution fitting, and visualization for reproducible risk modeling.

# Folder Structure
```graphql
student_t_var_cvar/
│
├── data/
│   ├── raw/
│   │   └── btc-usd-max.csv        # Raw $BTC returns
│   │
│   └── output/
│       └── data.csv               # Log Returns 
│
├── src/
│   ├── returns.py                 # Takes /raw and /output log returns 
│   ├── distributions.py           # Distribution fitting and risk metrics
│   └── visualization.py           # Plots (PDF, QQ, and density)
│
└── README.md                      # Project documentation
```
# Pipeline
## `returns.py`
Responsible for data ingestion and outputs log returns.

## `distributions.py`
Fits normal and student-t distributions to log retuns. 

## `visualization.py`
Returns distributions with fitted Normal & Student-t overlay, QQ-plots against each distribution, and left-tail density comparison to highlight extreme risk behavior. 
