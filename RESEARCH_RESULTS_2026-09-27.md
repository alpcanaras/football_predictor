Initial research run: `20260927T200604Z-387dcf43`.

This run rebuilt 11,105 matches from the raw English Premier League,
Bundesliga, Spanish La Liga, Italian Serie A and Turkish Super Lig files.
The comparison covers 848 matches with complete pre-closing reference odds in
the 60-day windows starting August 1 in 2024, 2025 and 2026. The 2026 window
is incomplete because the source currently ends in September. This is a new
experimental feature/model recipe, not a rerun of the existing production
ensemble. Historical periods remain exploratory, not prospectively held out.

| Forecast | 2024 log-loss | 2025 log-loss | 2026 log-loss | Overall |
|---|---:|---:|---:|---:|
| Market | 0.9525 | 0.9624 | 0.9777 | 0.96352 |
| Calibrated market | 0.9492 | 0.9607 | 0.9795 | 0.96231 |
| Football only | 0.9751 | 1.0051 | 1.0067 | 0.99485 |
| Market plus residual correction | 0.9492 | 0.9624 | 0.9777 | 0.96233 |

The residual model's improvement over the market is 0.00119 log-loss, with a
95% weekly block-bootstrap interval of [-0.00098, +0.00345] across 21 observed
weeks. The calibrated market's improvement is 0.00121 with interval
[-0.00117, +0.00383]. Neither establishes a repeatable improvement. The
football-only model is worse by 0.03133, with its improvement interval entirely
negative: [-0.04473, -0.02071].

The residual model's unit-stake return at historical Bet365 snapshot prices
was +1.65% on 101 hypothetical selections under the fixed 5% estimated-return
threshold. **This does not demonstrate a betting edge:** quote timestamps,
actual availability, price changes and costs are unverified, and the prediction
improvement is within uncertainty. Do not use that ROI as a deployment reason.

The next useful evidence is from frozen prospective forecasts and additional
predeclared chronological windows. There is no basis here to replace the
production ensemble or increase financial exposure. The new pipeline makes
that decision inspectable rather than assuming a favourable score is real.

Full predictions, per-league/cohort metrics, fitted artifacts and the data/code
provenance manifest are in `reports/research/20260927T200604Z-387dcf43/` locally.
Run the command in `RESEARCH.md` to reproduce the experiment; output artifacts
are intentionally excluded from git, while this measured summary is tracked.
