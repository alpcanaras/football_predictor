# V2 models — results (2026-09-30)

**Status (2026-09-30): v2 serves the Turkish 1X2; everything else is still
v1 by default.** The app has a model switch in its sidebar:

| mode | 1X2 served from |
|---|---|
| **Auto** (default) | v2 for the Turkish Süper Lig, v1 everywhere else |
| **v1** | the production models (`models/tier1/`) everywhere |
| **v2** | the frozen v2 bundle everywhere |

With bookmaker odds, every mode uses the market. The Match, Fixtures and
Toto tabs show v1 and v2 side by side. The switch covers the 1X2 and the
goal markets (over/under 1.5/2.5/3.5, BTTS, xG; see "Goals model" below).
Half-time markets stay v1.

Why Turkey went first: the Turkish v1 models are overfit. They score 0.67
log-loss on a season they trained on, but 1.19–1.25 on the unseen 2026/27
matches, which is worse than 1/3-1/3-1/3. On those same matches the market
scored 1.02 and v2 1.03. Serving v2 there early was the owner's call.

Production v1 is untouched: all 627 v1 model files match
`models/_backup/prod_checksums_2026-09-29.txt`, and git tag `prod-2026-09-29`
is the durable copy. The frozen v2 bundle is `models/v2/2026-09-30_sel_ens_mkt3/`.
`scripts/selftest.py` checks that v2 serves every league and that the switch
routes as documented.

## What v2 is

1. **One feature engine** (`scripts/research_features.py`) for both training
   and live prediction, so the two cannot drift apart. Features for a day are
   computed before that day's results or prices enter the state.
2. **Market distillation.** Training targets mix the result with the
   Shin-de-vigged average closing price (`alpha` ≈ 0.87 after tuning).
   Closing prices are used only as targets for finished matches, never as a
   feature of the match being predicted.
3. **Market ratings.** A second rating per team, fitted to the prices of its
   *finished* matches (average close → Pinnacle → Bet365 → pre-match), plus
   its recent market draw price and points-vs-market residual. It carries
   the market's opinion of a team into fixtures that have no price.
4. **Tuned XGBoost + LightGBM, averaged, then one temperature**
   (`scripts/v2_train.py`, `V2Model`). The backtest scores exactly the
   object the bundle serves.

## How it was tested

- **Choose** on four 6-month tuning windows in 2022–23 (24,801 matches).
  The search could not see anything from 2024-01-01 onwards.
- **Confirm once** on six 6-month test windows, 2024-01 → 2026-09
  (33,101 matches).
- Each window: fit on a trailing lookback ending 90 days before the window,
  set the temperature on those 90 days, predict the window.
- Every comparison is paired, with 95% weekly block-bootstrap intervals,
  because same-weekend matches are not independent.
- Leakage checks: unit tests (`tests/test_research.py`, 25 pass), plus a
  real-data sabotage test. Randomising every price and result on
  2023-10-21 changed 0 of 162 feature rows that day and 0 of 63,789 before
  it, and changed 734 of 864 after it.

## Results — log-loss (lower is better), gain vs the research baseline

| step | tuning 2022–23 | test 2024–26 |
|---|---|---|
| research baseline | 1.01576 | 1.02389 |
| + distillation | +0.00081 [+0.00033, +0.00128] | +0.00114 [+0.00075, +0.00154] |
| + market ratings (first version, k=40) | +0.00869 [+0.00703, +0.01028] | +0.00833 [+0.00716, +0.00951] |
| **chosen: k=260 ratings, tuned XGB+LGBM** | **+0.01333 [+0.01122, +0.01527]** | **+0.01253 [+0.01101, +0.01402]** |

On test, the chosen model is +0.00420 [+0.00319, +0.00522] better than the
first market-rating version. Top-2 hit rate on test is 0.7598 → 0.7709.

### Head-to-head vs the production models (`scripts/v2_compare.py`)

This uses only matches played after each production model was written, and
v2 is trained "as of" the same date.

| pool | matches | production | v2 | gain [95% CI] | accuracy | top-2 |
|---|---|---|---|---|---|---|
| per-league models (since 2026-03-24) | 3,998 | 1.04609 | 1.01570 | **+0.0304 [+0.0234, +0.0370]** | 46.9% → 49.6% | 74.1% → 75.6% |
| served model: per-league + pooled (since 2026-08-22) | 1,611 | 1.03819 | 1.01465 | **+0.0235 [+0.0124, +0.0311]** | 47.8% → 49.2% | 73.8% → 75.5% |

### Against the market

- On the 2,120 head-to-head matches with pre-match odds: market 1.00605,
  v2 1.00773 (−0.0017 [−0.0071, +0.0038]), production 1.04654.
- Over the full 2024–26 test (20,388 priced matches), v2 alone is
  −0.0047 [−0.0062, −0.0033] behind the market.
- Blending v2 into the market: the tuning windows pick 80/20, but it does
  not hold on test (−0.0002). **With odds, use the odds.** v2 is for
  fixtures without them.

## Chosen recipe (`reports/v2/selected_recipe.json`, also in the bundle's meta.json)

| member | alpha | half-life | lookback | model |
|---|---|---|---|---|
| XGBoost | 0.867 | 2000 d | 4 y | 227 trees, depth 7, lr .049, min_child_weight 43 |
| LightGBM | 0.867 | 500 d | 4 y | 863 trees, 15 leaves, lr .023, bagging on |

Market ratings: Elo-style update, k=260, season carry-over 0.9, home
advantage 60 (`research_features.MKT_*`, version `causal-day-state-v1+mkt3`).

## Goals model (`scripts/v2_goals.py`, bundle `models/v2/2026-09-30_goals_poisson_tuned/`)

- **What it is:** two Poisson rate models (home goals, away goals) on the
  same causal features. It also uses each team's recent over/under-2.5
  prices (`FeatureState(totals=True)`, read only once a match is over),
  plus a Dixon-Coles low-score correction and a goal-level scale set on the
  calibration window.
- **Why one grid:** every goal market is read off one grid of scorelines,
  so the markets cannot contradict each other.
- **Poisson vs one classifier per market (like v1):**
  - Tuning windows: better on all four markets, clearly on over 1.5 and BTTS.
  - Test windows: better on 3 of 4 markets. Over 2.5 is a tie (−0.0004, n.s.).
- **Inputs, tuning windows:** market ratings help every market
  (+0.0009..+0.0023). Past over/under prices add +0.0002..+0.0005 on top.
- **Tuning (60 trials, tuning windows only):** the goals model prefers all
  seasons equally weighted, unlike the 1X2. Tuned vs untuned, confirmed on
  test: +0.0005..+0.0006 on every market, all CIs above zero.

**Head-to-head vs production v1** (4,076 matches unseen by v1):

| market | v1 | v2 | gain [95% CI] |
|---|---|---|---|
| over 1.5 | 0.5205 | 0.5141 | **+0.0064 [+0.0029, +0.0101]** |
| over 2.5 | 0.6844 | 0.6757 | **+0.0087 [+0.0049, +0.0124]** |
| over 3.5 | 0.6261 | 0.6168 | **+0.0093 [+0.0046, +0.0135]** |
| BTTS | 0.6831 | 0.6796 | +0.0035 [−0.0002, +0.0068] |

**Over 2.5 against the market:**
- The market is still better: v2 is −0.0044 on 2,197 priced matches.
- The bundle carries its own pool with the over/under odds (logit weights:
  model 0.118, book 1.106, fitted on the tuning windows). It ties the
  market on test, and beats reusing v1's weights by +0.0011.
- v1's current app blend is itself fine against the market alone
  (+0.0008, n.s.), so v1's anchoring is unchanged.

## Calibration check (what the Toto numbers rest on)

Test windows 2024–26, 33,101 matches:
- **Top pick:** when v2 says 45% it happens 44.3%; 64% → 63.7%;
  84% → 84.2%. Every bin is within its CI.
- **Draws:** 25.6% predicted, 26.3% happened.
- **Whole coupons,** 20,000 same-weekend coupons each:
  - Spor Toto 15 matches, 12+: P(prize) said 1.82%, happened 2.08%.
  - 13er Wette 13 matches, 10+: said 4.74%, happened 4.90%.
  - Slightly conservative, as expected when upsets cluster on a weekend.

## Tried and dropped

- Team over/under-price features (rich leagues): −0.00013 on tuning.
- Refitting the trees on the last 90 days too: worse on tuning (1.00294 vs 1.00263).
- Price-inversion rating rule: tied with k=260 (1.00320 vs 1.00312),
  including on teams with fewer than 8 priced games.
- Blending into the market: see above.

## Caveats

- I looked at the test windows several times during development. Every
  decision was made on the tuning windows, but repeated looking can still
  flatter the numbers slightly. The clean test is shadow mode.
- The head-to-head pools are small (7–28 weeks).
- Much of v2's gain comes from the market's own opinion. That makes it close
  to the market, but not independent of it.

## Shadow mode

```bash
python scripts/fetch_latest.py --apply --refresh-processed   # latest results first
python -m scripts.v2_bundle shadow      # log v2 / production / market for the next 7 days
python -m scripts.v2_bundle grade       # score everything logged that has been played
python -m scripts.v2_bundle status
```

The log is `reports/v2/shadow/shadow_log.csv`. It keeps the latest pre-match
prediction per fixture, and only fixtures from today onward are logged.

## Promotion checklist (the rest of the leagues — owner's call)

1. 3–4 league weekends of shadow logging. `grade` shows v2 vs v1 with a CI
   above zero.
2. ~~Wire v2 into `predict.py` / `toto._model_probs` behind a flag, with v1
   as the fallback~~ — done: `predict.MODEL_MODES` and the sidebar switch.
   Promoting everywhere means making `v2` the default mode, or growing
   `predict.V2_AUTO_LEAGUES`.
3. Keep the market-first policy for priced fixtures (`blend_weights.json`).
4. Re-run `scripts/selftest.py`, then tag the new production state.
