Broader rolling evaluation completed on 28 September 2026.

Run ID: `20260928T051657Z-30c77dc7`. The same fixed research recipe was run
with all 38 leagues available for training, rebuilding 99,814 raw matches.
Six non-overlapping, 180-day test windows begin on January 1 and July 1 of
2024, 2025 and 2026. The last window is incomplete. Each fold trains new
football and residual classifiers and fits its calibration on separate earlier
dates. No production model is changed.

Probability comparisons use the same 20,347 test matches with complete
pre-closing reference odds. The 38 training leagues do not all have those odds;
closing-only observations are deliberately excluded from market comparisons.
This expands the initial season-start study rather than changing its recipe.

| Forecast | Overall log-loss | Accuracy | Improvement versus market, 95% weekly bootstrap interval |
|---|---:|---:|---|
| Market | 1.001394 | 50.50% | Reference |
| Calibrated market | 1.001146 | 50.50% | +0.000248 [-0.000454, +0.000977] |
| Football only | 1.018486 | 48.80% | -0.017093 [-0.019249, -0.014754] |
| Market plus residual correction | 1.001328 | 50.43% | +0.000065 [-0.000335, +0.000501] |

The intervals resample 120 observed weeks. Neither market calibration nor
football-based residual correction establishes a reliable gain. The independent
football model remains weaker on average in this experiment. This does not
prove that other features or algorithms cannot help; it does mean extra compute
on this unchanged recipe is not presently justified as a route to an edge.

At the fixed 5% estimated-return threshold, the residual strategy's hypothetical
Bet365 snapshot return was -6.68% across 723 selections. These historical prices
have no established availability timestamps and costs are excluded, so this is
not an executable betting backtest. The calibrated-market variant's positive
return came from only eight selections and must not be interpreted as evidence
of profitability. No model is promoted on the basis of this run.

Separately, the all-league prospective challenger bundle was saved in
`reports/research/20260928T051457Z-9f069bfb/bundle.joblib`. Its football base model
fits 46,502 matches from 2022-09-29 through 2026-06-28, and calibration uses
2,555 later matches through 2026-09-24. The market residual base model fits the
29,337 corresponding training matches with pre-closing reference odds, with
1,286 calibration matches. The exclusive outcome cutoff is 2026-09-28.

The first prospective recorder invocation archived the downloaded fixture feeds
but produced zero forecasts: on September 28 the feed had no matches strictly
after the current UTC date. This is expected under the conservative date-based
eligibility rule, not evidence of a prospective trial already completed.

Next experiments should change an explicit information or modelling hypothesis
and remain separately versioned. Repeated selection on these reported windows
is research, not a new untouched test. Production models and existing scripts
remain available and unchanged.
