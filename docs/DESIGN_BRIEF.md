# Design brief: Crypto Risk Engine (jvick1/crypto-risk-engine)

> **Tier: PUBLIC (public GitHub repo).** An educational quant reference: CLI and CSV output, no UI, no trading signals.
> **Status: DRAFT for owner review.** Basis: the docs in this folder (PRD, TRD, APP_FLOW, SCHEMA, ROADMAP) as of `b464f75`, plus `src/visualization.py`. Nothing here is implemented yet.

## Audience
- **Primary:** Jake as a quant/risk practitioner.
- **Secondary (public):** reviewers, hiring managers and learners who want a small, readable VaR/CVaR reference on real crypto data.

## Brand fit
- **The best mission fit of any repo.** The brand book says Cypher-Fi exists for "raising awareness around cryptocurrencies… helping provide resources to others starting their crypto journey". A clear, honest crypto tail-risk tool is exactly that.
- It is public today, so how it looks reflects on Jake and on Cypher-Fi.
- Open question: should it carry Cypher-Fi branding at all? It lives under Jake's personal GitHub, next to his résumé repos.

## Look and voice
- **README voice:** teacher-like and precise.
  - Explain each method in one or two plain sentences ("Student-t has fatter tails, so it expects big drops more often than a Normal curve does").
  - Show the formula, then the plot.
  - Always say "historical" and "educational".
- **Plots (matplotlib):** move from matplotlib defaults to the shared chart conventions.
  - Today: `tab10` (`#1f77b4`/`#ff7f0e`/`#2ca02c`), "red"/"green" VaR lines, DejaVu font.
  - **Method colors, fixed across every plot:** Normal `#002366` · Student-t `#009fff` · Historical `#8c8c8c` (dashed). Bayesian (Phase 3) `#6d8bdc` with a shaded credible band in `#99d9ff`.
  - Actual returns: black at 50% alpha (already done). Breaches: a small ✖ marker in the method's color, not red dots. A breach is an event, not an alarm.
  - Font: Roboto registered from a TTF (copy Market Decoder's `_load_fonts()` pattern).
  - Caption on every saved figure: `BTC-USD daily · Yahoo Finance via yfinance · <start>–<end> · Educational, not investment advice`.
- **CLI/console:** plain `print` is fine. Put a one-line disclaimer at the start of the printed summary, and use fixed decimals (VaR/CVaR as percent with 2 decimals, `−4.87%`).

## Report / output conventions
- **CSV names:** `{coin}-backtest-results.csv` and `{coin}-backtest-summary.csv` are already kebab and consistent. Keep them. A shared-convention date suffix is optional (not needed for a reference repo).
- **Summary table columns:** method · breaches · expected · breach rate · Kupiec p · Christoffersen p · CC p · reject@5%.
  - Display "reject" as a word, not a color. If colored: reject = warn amber, pass = neutral. These are model-fit verdicts, not trading signals.
- **Reported results:** only quote numbers from runs that were actually done. The PRD's reference point (Nov 4 BTC run: 241 breaches, Kupiec p 0.026) is a real run and fine to cite with its date and settings.

## Status colors
Not a status tool. Use the shared states only for the test verdict (pass = neutral/ok, reject = warn) and for fit fallbacks (Student-t → Normal fallback = warn, already a `RuntimeWarning`).

## What it should feel like
A clean textbook chapter with working code: calm blues, one idea per chart, honest about where the models fail.

## Do / Don't
- **Do:**
  - Keep "Educational project, not investment advice" at the top and bottom of the README.
  - Explain that `alpha` is a tail probability.
  - Show where Student-t fails the test, too (the Phase 2 note already does this).
- **Don't:**
  - Use the words signal, buy, sell, or "safe to hold".
  - Use red/green "good/bad" coloring.
  - Make performance claims.
  - Put Cypher-Fi logos in a personal repo without Jake's OK.
  - Show screenshots of yfinance data without the source line. **Flag:** public display of Yahoo-sourced data; check the Yahoo terms (site review L2).

## Open questions for Jake
1. Should this public repo say "A Cypher-Fi project" (with `© Cypher-Fi, LLC` and a LICENSE), or stay personal/portfolio? There's no LICENSE file today, so others can't legally reuse it.
2. OK to recolor the plots to the brand method colors (Normal navy, Student-t cerulean, Historical grey)?
3. Is this a candidate for a Cypher-Fi Labs YouTube episode ("why crypto crashes break normal-curve risk models")?
