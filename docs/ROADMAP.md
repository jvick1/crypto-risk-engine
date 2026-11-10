# Roadmap

Educational project — not investment advice. Each phase lands on its own feature branch, is reviewed, then merged to `main`.

| Phase | Scope | Status | Done when |
|---|---|---|---|
| 0 | Cleanup: remove committed `__pycache__`, `.gitignore`, pinned `requirements.txt`, docstring/help fixes, `data/backtest` auto-create, `--no-verbose`, pytest skeleton, README install/flags | **Done** — merged in PR #1 (`ae194cb`) | Merged to `main`; tests pass |
| 1 | Kupiec POF (chi2 df=1), Christoffersen independence (df=1) and conditional coverage (df=2) tests; per-method summary | **Done** — pushed (`feat/phase1-backtest-tests`), awaiting review | `--backtest` prints and saves `data/backtest/{coin}-backtest-summary.csv` with breaches, expected, breach rate, LR stats, p-values, reject-at-5% flags; edge cases (zero breaches, no transitions) return finite values; hand-checked unit tests pass |
| 2 | Student-t robustness: bound df (nu > 2), fit fallback, CVaR guard | **Done**: pushed (`feat/phase2-student-t-guards`), awaiting review. Backtest effect (BTC, alpha 0.05, 252-day window): the 2.01 df floor binds in 1,064 of 4,179 rolling windows (raw fits as low as 1.23); Student-t breaches went from 237 to 241 and the Kupiec p-value from 0.051 (pass) to 0.026 (reject at 5%). Full-sample VaR/CVaR unchanged. Floor kept by design (df <= 2 means infinite variance). | Fitted df always > 2; a failed/degenerate fit falls back (e.g. to Normal) with a warning; CVaR never diverges or returns NaN/inf; tests cover each case |
| 3 | PyMC Bayesian Student-t on the full sample, `--methods bayes` | Planned | Weakly informative priors with nu > 2; VaR/CVaR from the posterior predictive with credible intervals; R-hat and ESS reported (warn if R-hat > 1.01); seeded and reproducible; tests pass |
| 4 | Rolling Bayesian backtest | Planned | Refit every k days (configurable), warm start from previous posterior, conjugate Normal-Inverse-Gamma fast path, cached fits; output feeds the Phase 1 summary; runtime documented |
| 5 | README update | Planned | README documents all methods, flags, outputs and caveats, and matches the code |

## Backlog (unscheduled)
- `--ticker` support for stocks (ticker currently forced to `{COIN}-{VS}`)
- EWMA / GARCH volatility models
- CVaR backtest (e.g. Acerbi–Székely)
- Faster backtest (vectorize / parallelize Student-t refits)
- CI (GitHub Actions running pytest)
