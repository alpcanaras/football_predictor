The research pipeline is isolated from the current application and production
models. Existing scripts remain available. New runs write unique directories
under `reports/research/`; they never overwrite an earlier run or a production
model. Work is on branch `research/season-aware-evaluation`.

Run the behavioral checks:

```bash
./venv/bin/python -m unittest discover -s tests -v
```

Run the initial, fixed-recipe comparison across five leagues and the opening
60-day windows of the 2024, 2025 and 2026 seasons:

```bash
./venv/bin/python -m scripts.research_train evaluate
```

Change the scope explicitly, for example:

```bash
./venv/bin/python -m scripts.research_train evaluate \
  --leagues british_pl german turkish \
  --cutoffs 2024-08-01 2025-08-01 2026-08-01 --test-days 60
```

Each fold fits on a trailing four-year window ending 90 days before the test,
then calibrates on those separate 90 days. Complete dates stay together. Elo
uses declared fixed settings, not the production parameters tuned on the full
dataset. Every fixture on a day is featurized before any result that day updates
the shared state. Future fixture forecasts use the same feature function.

The four comparisons are normalized market probabilities; market probabilities
with a calibration temperature; a football-only pooled XGBoost model; and an
XGBoost model using market log probabilities as its base margin. The latter
learns corrections rather than reconstructing the entire probability from
scratch. Its correction strength is selected on calibration matches, including
zero as an option. Football and market calibration are also fitted only before
the test. Tree hyperparameters are fixed in this initial recipe, not optimized
against the reported test scores.

Reports include paired comparisons on the same matches, weekly block-bootstrap
intervals, early-season and top-pick-disagreement cohorts, and per-league
breakdowns. Positive paired improvement means lower loss than the market.
These subgroups are descriptive; choosing a strategy from them requires another
later test. The current incomplete 2026 window reports only observed results.

Historical odds use complete pre-closing triplets only; closing odds never
substitute for missing pre-closing observations. Reference source columns are
saved with predictions. Hypothetical unit-stake returns use Bet365 pre-closing
columns with a fixed 5% estimated-return threshold, at most one selection per
match. **These are not executable backtest returns:** historical price timestamps,
price availability and costs are not established. Same-day features may also
include information learned after the historical price was collected. They
must not be presented as a demonstrated betting edge.

Train an isolated challenger for prospective recording:

```bash
./venv/bin/python -m scripts.research_train train --cutoff 2026-09-27
```

Use the resulting `bundle.joblib` only for predictions on/after its cutoff.
Bundles carry fit/calibration dates, feature schema and version, Elo settings,
training counts, hyperparameters and fitted calibration settings. The adjacent
manifest carries raw-file hashes, source-code hashes, git revision, arguments
and library versions. Artifact prediction rejects earlier dates and unknown
leagues. Regenerated features must use the bundle's Elo settings.

Current limitations: team identities are scoped to a league, so promoted teams
start from a league prior and have explicit history counts. Source seasons are
used where available; fixture fallback seasons use calendar years for declared
calendar leagues and July boundaries otherwise. Split competitions such as
Mexico need a richer stage model before treating standings as stage-specific.
No lineup, transfer, player-availability or shot-quality xG data is included.
Probability benchmarking is exploratory; prospective results are required.

Archive prospective forecasts from a trained bundle:

```bash
./venv/bin/python -m scripts.research_record record \
  --bundle reports/research/RUN/bundle.joblib --refresh
./venv/bin/python -m scripts.research_record grade \
  --record reports/research_records/RECORD
```

Each recording gets a unique directory containing the original fixture feeds,
their hashes and download timestamps, model hash, raw-result hashes and the
forecast probabilities. Only fixtures strictly after the current UTC date are
included, because the feed's kickoff timezone is not established. All state
updates precede that date. Grading matches the exact league/date/home/away
identity and writes a new grading file, preserving the original forecast.
Download time is explicitly distinct from the unknown source quote time. These
archives are local and are not scheduled automatically.

`scripts/toto_settlement.py` provides the new settlement foundation without
changing the existing application. `tier_counts` counts every purchased column
in every exact prize tier. `settle` uses published payouts per winning column
and deducts costs/fees. `scenarios` models independent public columns using
supplied crowd percentages and field size; `evaluate` estimates net money
using explicit tier pots, counting own winning columns in each pot's denominator.
It supports comparing candidate systems on common simulations, but is not yet
wired into the app's optimizer. Public-ticket independence is an assumption,
and jurisdiction-specific rollover rules must be supplied through the pots.
