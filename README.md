# Football Predictor

Per-league XGBoost + LightGBM ensembles for 1X2, Over/Under (1.5/2.5/3.5),
BTTS, Half-Time and xG markets across 28 leagues. All historical match data
comes from [football-data.co.uk](https://www.football-data.co.uk). Models are
calibrated (isotonic / Platt chosen per market) and averaged across engines at
prediction time.

## Layout

```
data/<league>/*.csv           Raw CSVs per league (one file per season + cumulative "new" files)
data/_incoming/<date>/        Staging area for scripts/fetch_latest.py
full_processed_data.csv       Features-engineered matches (regenerated from data/)
models/tier1/                 Production models: model_<market>_<league>[_lgbm].joblib
models/elo_params.json        Tuned Elo hyperparameters
scripts/                      Data loading, training, prediction, validation
launchd/                      Dormant LaunchAgent plist for daily auto-fetch
Predictor_latest.ipynb        Interactive prediction + analytics notebook
```

## Everyday use

**Easiest: double-click `run.command`** (macOS) — it sets up the environment
on first run and opens the app in your browser. Nothing else to install.

Or from a terminal:

```bash
pip install streamlit      # one-time
streamlit run app.py       # opens in your browser
```

Four tabs: pick a league + two teams for a full prediction card; browse all
upcoming fixtures anchored to live odds; see the World Cup slate; or build a
**Toto coupon** (the 🎟️ tab). Replaces the notebook for everyday use.

### Toto coupon optimizer

For pools games (Turkish Spor Toto: 15 matches, prize 12+; German 13er Wette:
13 matches, prize 10+). Enter the week's matches and bookmaker 1X2 odds; it
returns calibrated probabilities, your single-column chance of hitting each
prize tier, and — given a column budget — the optimal **system play** (which
least-predictable matches to cover with doubles/triples). The maths is exact
(Poisson-binomial over independent matches); coverage is allocated greedily by
threshold-probability gained per column spent.

Where a row has no odds the model fills the probabilities in: **club** teams via
the league ensembles, **national** teams (World Cup weeks) via the international
Elo+Poisson model — so a coupon of `turkey`/`paraguay`/… is priced properly
instead of defaulting to 1/3 each. In the app you can also push a match straight
from the 🎯 Match tab into the coupon with **➕ Add to Toto**.

```bash
python scripts/toto.py --template                              # blank coupon.csv
python scripts/toto.py --coupon coupon.csv --game turkish --budget 64
python scripts/toto.py --coupon coupon.csv --game german  --budget 48
```

Key fact the tool exploits: both games allow exactly **3 misses** (12/15 and
10/13), so difficulty is about per-match predictability — German coupons draw
from harder lower divisions, which is why 10/13 is tougher than 12/15. Spend
coverage on the toss-ups, bank the favourites.

**One command for everything coming up** (club fixtures with live odds
anchoring + today's World Cup matches):

```bash
python scripts/today.py              # next 3 days
python scripts/today.py --days 7 --csv picks.csv
```

It auto-downloads the bookmaker odds feed for upcoming fixtures (free, no API
key), predicts every match it has models for, anchors 1X2 and O/U 2.5 to the
market, and flags the model-vs-market edge per match.

For a single match or interactive browsing:

```bash
python scripts/predict.py --home "Flamengo RJ" --away "Coritiba"
python scripts/predict.py            # interactive mode
```

Or open [Predictor_latest.ipynb](Predictor_latest.ipynb). Pick a league (or keep
"All leagues") and two teams from the dropdowns; the prediction card refreshes
automatically. The analytics cells at the bottom of the notebook evaluate the
saved ensemble on the last 90 days of data — top-2 1X2 accuracy, per-market
Acc/LogLoss/Brier, reliability plots, value-bet scanner, and xG MAE.

Flip `refresh_data = True` in the first cell if you dropped new CSVs into
`data/<league>/` and want to rebuild `full_processed_data.csv`.

Weekly data refresh (results catalogues update after matchdays):

```bash
python scripts/fetch_latest.py --apply --refresh-processed
```

## Refresh + retrain workflows

```bash
# Rebuild full_processed_data.csv from whatever is under data/<league>/
python scripts/refresh_data.py

# Retrain a single (league, engine) combination without touching anything else
python scripts/train_one.py --league CH --engine lgbm

# Full retrain (hours)
python scripts/train.py --fresh

# Validation report vs bookmaker
python scripts/validate.py --compare
```

## Daily auto-fetch (football-data.co.uk)

The fetcher downloads the current-season rich-league file
(`mmz4281/<season>/<div>.csv`) and the cumulative "new" file
(`new/<code>.csv`) for every league registered in
[`config.FETCH_SOURCES`](scripts/config.py).

```bash
# 1. Staging-only (default). Writes to data/_incoming/<date>/<league>/ and
#    leaves data/<league>/ untouched. Inspect the report to confirm row counts
#    and latest dates look reasonable.
python scripts/fetch_latest.py

# 2. Promote the latest staged batch into data/<league>/ and rebuild features:
python scripts/fetch_latest.py --apply --refresh-processed

# 3. Subset (useful for testing a single league):
python scripts/fetch_latest.py --only british_pl,italian --apply
```

### Activating the daily LaunchAgent

The plist is **not** loaded by default. Once you are happy running the script
manually for a few days, activate it:

```bash
# Install into the per-user LaunchAgents directory (symlink keeps the repo as
# the source of truth, so edits to the plist propagate).
ln -sf "$(pwd)/launchd/com.footballpredictor.fetch.plist" \
       ~/Library/LaunchAgents/com.footballpredictor.fetch.plist

# Enable + start (runs daily at 07:00 local time).
launchctl load -w ~/Library/LaunchAgents/com.footballpredictor.fetch.plist

# Status / logs:
launchctl list | grep footballpredictor
tail -f logs/fetch.log logs/fetch.err

# Deactivate:
launchctl unload -w ~/Library/LaunchAgents/com.footballpredictor.fetch.plist
```

Adjust `Hour`/`Minute` in
[`launchd/com.footballpredictor.fetch.plist`](launchd/com.footballpredictor.fetch.plist)
if you want a different run time.

## League coverage

25 leagues, split into two format families:

- **Rich** (full columns: shots, corners, fouls, cards, HT results) → trains
  HT1X2 and HT O/U 0.5 in addition to the core markets:
  English PL / Championship / League One / Conference, German Bundesliga / 2.
  Bundesliga, Spanish La Liga / Segunda, Italian Serie A, French Ligue 1,
  Dutch Eredivisie, Belgian Pro League, Portuguese Liga, Greek Super League,
  Turkish Super Lig.
- **Sparse** (one file, results + odds only): Argentine Liga, Brazilian Serie
  A, Swiss Super League, Danish Superliga, Chinese Super League, Finnish
  Veikkausliiga, Irish Premier Division, Japanese J-League, Mexican Liga MX,
  Norwegian Eliteserien, Russian Premier League, Swedish Allsvenskan, USA MLS.

Summer coverage (European off-season): MLS, Liga MX, Irish Premier Division,
Brazil, Argentina, Japan, China, Norway, Sweden, Finland all run through the
European summer.

## Market-anchored predictions

`predict.py` now blends its 1X2 and O/U 2.5 probabilities with bookmaker
odds whenever the fixture appears in football-data.co.uk's upcoming-fixtures
feed (downloaded automatically to `data/_fixtures/`, cached 12h). Blend
weights live in `models/blend_weights.json` and were fitted out-of-sample by
[`scripts/blend.py`](scripts/blend.py) — on Feb–May 2026 data the
market-anchored blend scores 1.003 log-loss vs 1.033 for the model alone
(1X2), 0.68 vs 0.69 for O/U 2.5. Per-league weights were tested and did not
beat global weights (kept in `evaluate` for future re-checks). Refit after
each retrain:

```bash
python scripts/blend.py evaluate   # tune/eval split report (model vs book vs DC)
python scripts/blend.py fit        # refit weights on the full OOS window
python scripts/fixtures.py         # inspect the current fixtures+odds feed
```

[`scripts/dixon_coles.py`](scripts/dixon_coles.py) provides a time-decayed
Dixon-Coles baseline used as a third blend component (currently near-zero
weight; kept for leagues/periods without odds).

## Pooled cross-league model (1X2)

[`scripts/pooled.py`](scripts/pooled.py) trains one model over all 75k matches
with the league as one-hot features (partial pooling). Validated out-of-sample
(held out at 2026-02-01, 2,654 matches): per-league 1.0242, pooled 1.0157,
**50/50 blend 1.0149** log-loss — so `predict.py` now averages the per-league
1X2 ensemble with the pooled model. (O/U 2.5 was tested too; per-league won
there, so pooling is 1X2-only.) Retrain on all data after a refresh with:

```bash
python scripts/pooled.py train --markets 1x2          # production (no cutoff)
python scripts/pooled.py train --markets 1x2 --cutoff 2026-02-01  # + evaluate
python scripts/pooled.py evaluate                     # pooled vs per-league vs book
```

## International / World Cup module

National teams are handled by a separate model in
[`scripts/international.py`](scripts/international.py): a tournament-weighted
Elo (eloratings.net K-scheme, goal-margin multiplier, neutral-venue handling)
over `data/global/international.csv`, plus a Poisson goal model that converts
Elo differences into 1X2 / O-U / BTTS probabilities.

```bash
# Refresh results + upcoming WC fixtures from the martj42 GitHub dataset
python scripts/international.py update

# Current Elo top-30
python scripts/international.py ratings

# Single fixture (neutral venue by default)
python scripts/international.py predict --home Brazil --away Morocco

# Leak-free backtest on WC 2018, WC 2022, Euro 2024
python scripts/international.py backtest

# All WC 2026 group fixtures + Monte-Carlo advancement probabilities
python scripts/international.py wc2026
```

Backtest reference (group + knockout, pre-tournament data only): 54.2%
accuracy / 1.022 log-loss across WC 2018, WC 2022 and Euro 2024 — on par with
the club models. Knockout scores in the dataset include extra time, which
slightly biases the evaluation against draws.

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
