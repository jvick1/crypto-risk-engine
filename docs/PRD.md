# Product Requirements: Crypto Risk Engine

> Educational project, not investment advice. Status reflects the `feat/phase2-student-t-guards` branch (Phases 0-2 built; Phases 3-5 planned). See [ROADMAP.md](ROADMAP.md).

## Who it is for
- **Primary user:** the repo owner (a quant / risk practitioner) studying downside risk of crypto assets from the command line.
- **Secondary:** reviewers and learners who want a small, readable reference for VaR / CVaR estimation and backtesting on real market data.

## Problem
Crypto daily returns are fat-tailed and volatile. A Normal-distribution VaR tends to understate extreme losses, and a single VaR number says nothing about whether the model would have held up historically. The user needs a reproducible way to:
1. estimate VaR and CVaR for a coin with several methods side by side;
2. see visually how well each distribution fits the tails;
3. backtest each method out of sample and test the breaches statistically.

## Goals
- Pull full daily price history for any Yahoo Finance `{COIN}-{VS}` pair and compute daily log returns.
- Estimate VaR and CVaR at a user-chosen tail probability `alpha` with Normal, Student-t and Historical methods.
- Run a rolling-window, no-look-ahead backtest and judge it with Kupiec POF, Christoffersen independence and conditional-coverage tests.
- Keep the Student-t estimates numerically safe (finite variance, no diverging CVaR).
- Keep the code modular (one concern per module), tested, and documented so it matches the code.

## Non-goals
- Not investment advice, trading signals or portfolio recommendations.
- No live/intraday data, streaming, or real-time alerting (daily close only).
- No portfolio or multi-asset VaR (single asset per run).
- No web UI, API server or database; it is a local Python CLI writing CSV files.
- No order execution or exchange connectivity; no user accounts or secrets.

## Features

| Feature | Status | Where |
|---|---|---|
| Download full daily close history from Yahoo Finance (`{COIN}-{VS}` ticker) | Built | `src/api.py` |
| Daily log returns + summary stats (mean, std, excess kurtosis) | Built | `src/returns.py` |
| Normal (MLE) and Student-t (MLE) fits | Built | `src/distributions.py` |
| Student-t guards: df floor at 2.01, Normal fallback on a failed fit, CVaR guard for df <= 1 | Built (Phase 2, awaiting review) | `src/distributions.py`, `src/risk_metrics.py` |
| VaR / CVaR: Normal, Student-t, Historical | Built | `src/risk_metrics.py` |
| Diagnostic plots: PDF overlay, QQ plots, left-tail density | Built | `src/visualization.py` |
| Rolling-window backtest (default 252 obs) with breach flags, saved to CSV | Built | `src/backtest.py` |
| Kupiec POF, Christoffersen independence and conditional-coverage tests; per-method summary CSV | Built (Phase 1, awaiting review) | `src/backtest_tests.py` |
| Rolling VaR plot with breaches | Built (with `--backtest`) | `src/visualization.py` |
| Bayesian Student-t (PyMC) VaR / CVaR with credible intervals, `--methods bayes` | **Planned** (Phase 3) | - |
| Rolling Bayesian backtest (refit every k days, warm start, conjugate fast path, cache) | **Planned** (Phase 4) | - |
| README refresh covering all methods | **Planned** (Phase 5) | - |
| `--ticker` for stocks, EWMA/GARCH, CVaR backtest, faster backtest, CI | **Backlog** (unscheduled) | - |

## Success criteria
- `python -m pytest` passes (26 tests on the current branch).
- `python -m src.main --coin_symbol btc` completes end to end on live Yahoo data and prints Normal and Student-t VaR / CVaR.
- `--backtest` writes `data/backtest/{coin}-backtest-results.csv` and `{coin}-backtest-summary.csv`, with no look-ahead (each forecast uses only the prior `window_size` returns).
- Fitted Student-t df is always >= 2.01; CVaR is never NaN/inf for valid inputs.
- Backtest results are interpreted with the statistical tests, not only the raw breach rate. Reference point (11/4 BTC run, alpha 0.05, window 252): after the Phase 2 df floor, Student-t had 241 breaches and a Kupiec p-value of 0.026 (reject at 5%); see [ROADMAP.md](ROADMAP.md).
- Docs (this PRD, [TRD](TRD.md), [App flow](APP_FLOW.md), [Schema](SCHEMA.md), [Roadmap](ROADMAP.md)) match the code; anything not built is marked planned.
