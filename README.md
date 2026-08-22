# Football Predictor

Per-league XGBoost + LightGBM ensembles for 1X2, Over/Under (1.5/2.5/3.5),
BTTS, Half-Time and xG markets across 38 leagues. All historical match data
comes from [football-data.co.uk](https://www.football-data.co.uk). Models are
calibrated (isotonic / Platt chosen per market) and averaged across engines at
prediction time.

Built for pools play (Turkish Spor Toto, German 13er Wette), where the opponent
is the crowd rather than a sharp bookmaker — so the goal is **calibrated**
probabilities and good coverage allocation, not beating the market. Run
`python scripts/selftest.py` after any refresh or retrain.

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

Four tabs, Toto first because that is the weekly job: build a **coupon**;
browse **fixtures** anchored to live odds; get a full prediction card for a
single **match**; or see upcoming **internationals**. Replaces the notebook for
everyday use. The sidebar shows data freshness, flags stale leagues, and
refreshes everything from the internet in one click without a restart.

### Toto coupon optimizer

For pools games (Turkish Spor Toto: 15 matches, prize 12+; German 13er Wette:
13 matches, prize 10+). Enter the week's matches and bookmaker 1X2 odds; it
returns calibrated probabilities, your single-column chance of hitting each
prize tier, and — given a column budget — the optimal **system play** (which
least-predictable matches to cover with doubles/triples).

The maths is exact: a Poisson-binomial over independent matches (verified
against a 200k-run Monte Carlo). Coverage is chosen by seeding with a
marginal-gain greedy pass and then hill-climbing over single-match and
pairwise changes, which matches brute-force optimum on every instance small
enough to enumerate. Plain greedy under-spends the budget badly — it would
leave 27 of 48 paid-for columns unused — so on a 13-match German coupon this
lifts P(≥10) from 21.7% to 24.4% at a budget of 64, for the same money.

Where a row has no odds the model fills the probabilities in: **club** teams via
the league ensembles, **national** teams (Nations League, qualifiers, friendlies)
via the international model — so a coupon of `turkey`/`paraguay`/… is priced properly
instead of defaulting to 1/3 each. Name matching is accent-folded, so `curacao`,
`besiktas`, `koln` resolve to `Curaçao`, `Beşiktaş`, `Köln`.

In the app the Toto tab keeps **two separate coupons** (Turkish + German),
entered as pasted `Home - Away` lines (one per match, optional `o1 ox o2` after).
They're saved to `data/_toto/<game>.txt`, so they **persist across reloads and
browser tabs** until you clear them. Any match in the other tabs can be pushed
into either coupon with the **➕ Add** control, and duplicates are caught
(order-insensitive and accent-folded) rather than silently double-counted.

Three things that make building a coupon quick:

- **📡 Fill odds from feed** — looks every odds-less row up in the live
  upcoming-fixtures feed and fills in the 1X2 odds. Since the market dominates
  wherever odds exist, this is the single most valuable thing you can do to a
  coupon, and it saves typing them by hand.
- **Did you mean…?** — an unrecognised name gets a dropdown of close matches
  and a one-click Fix that rewrites the line. Names are resolved through an
  alias layer (`psg`, `bvb`, `gladbach`, `atleti`, `spurs`, …) plus accent
  folding, applied at lookup time — the source CSVs are never rewritten,
  because the fetcher replaces them on every refresh.
- **💸 What does another column buy me?** — runs the optimiser across budgets
  so the diminishing return is visible before you choose a stake.

```bash
python scripts/toto.py --template                              # blank coupon.csv
python scripts/toto.py --coupon coupon.csv --game turkish --budget 64
python scripts/toto.py --coupon coupon.csv --game german  --budget 48
```

Key fact the tool exploits: both games allow exactly **3 misses** (12/15 and
10/13), so difficulty is about per-match predictability — German coupons draw
from harder lower divisions, which is why 10/13 is tougher than 12/15. Spend
coverage on the toss-ups, bank the favourites.

### Did it actually work? (coupon history)

Every tier probability rests on the per-match numbers being honest, so the app
can record what it predicted and grade it once the results are in — **💾 Save
to history** after an analysis, then **📈 History & calibration**.

```bash
python scripts/track.py list          # saved coupons
python scripts/track.py grade         # correct vs expected, threshold hit rate
python scripts/track.py calibration   # claimed confidence vs realised
```

The benchmark to watch is **correct vs expected**: expected is the
Poisson-binomial mean, so landing consistently below it means the
probabilities are optimistic rather than you being unlucky. As measured on
1,960 out-of-sample matches the club models are close to honest — top pick
claimed 48.8% and realised 46.7% (95% CI ±2.2), draws 25.8% predicted against
26.5% actual.

## Health check

```bash
./venv/bin/python scripts/selftest.py           # everything
./venv/bin/python scripts/selftest.py --quick   # skip the app boot
```

Verifies each league file really holds that league's matches, flags stale
leagues, checks no club is in two leagues at once, tests the Toto maths
against Monte Carlo and brute force, loads and predicts with every league's
models, fits the international model, and boots the Streamlit app against a
scratch coupon directory. Worth running after any data refresh or retrain.

**One command for everything coming up** (club fixtures with live odds
anchoring, plus any upcoming international matches):

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

In the app, the sidebar's **🔄 Update league data** does steps 2 and 3 in one
click and hot-reloads without a restart, reporting per-league outcomes.

**Payload validation (do not remove).** football-data.co.uk runs Apache with
MultiViews: when a season's file does not exist yet, the server may answer
with a *similar* file and HTTP 200. This is not hypothetical — `EC.csv`
(English Conference) was served in answer to `E0.csv` and written into
`data/british_pl/`, putting 12 Conference matches and 23 non-league clubs into
the Premier League. Every download is now checked to really be the league
requested (rich leagues by the URL's `Div` code, sparse ones by
`config.SPARSE_COUNTRY`), and a rejected payload is deleted from staging so it
can never be promoted. `SWE.csv` (Sweden) and `SWZ.csv` (Switzerland) are one
typo apart, so this matters beyond the case that triggered it. A season that
is genuinely not published yet now reports `not-published` and the existing
data is kept.

Statuses worth reading in the report: `OK`, `not-published` (season has not
started), `wrong-div:` / `wrong-country:` (payload rejected — investigate),
`payload-not-csv`.

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

38 leagues, split into two format families:

- **Rich** (full columns: shots, corners, fouls, cards, HT results) → trains
  HT1X2 and HT O/U 0.5 in addition to the core markets:
  English PL / Championship / League One / **League Two** / Conference,
  **Scottish Premiership / Championship / League One / League Two**,
  German Bundesliga / 2. Bundesliga, Spanish La Liga / Segunda,
  Italian Serie A / **Serie B**, French Ligue 1 / **Ligue 2**,
  Dutch Eredivisie, Belgian Pro League, Portuguese Liga, Greek Super League,
  Turkish Super Lig.
- **Sparse** (one file, results + odds only): Argentine Liga, Brazilian Serie
  A, Swiss Super League, Danish Superliga, Chinese Super League, Finnish
  Veikkausliiga, Irish Premier Division, Japanese J-League, Mexican Liga MX,
  Norwegian Eliteserien, Russian Premier League, Swedish Allsvenskan, USA MLS,
  **Austrian Bundesliga**, **Polish Ekstraklasa**, **Romanian Superliga**.

The leagues in bold were added because Toto coupons
draw on them heavily; without models those rows fell back to 1/3-1/3-1/3, which
is the worst possible input to a Poisson-binomial.

Summer coverage (European off-season): MLS, Liga MX, Irish Premier Division,
Brazil, Argentina, Japan, China, Norway, Sweden, Finland all run through the
European summer.

Adding a league: register it in `LEAGUE_REGISTRY` and `FETCH_SOURCES`, then

```bash
python scripts/fetch_latest.py --only <league> --backfill 7
python scripts/refresh_data.py
python scripts/train_one.py --league <league> --engine xgb   # then lgbm
```

## Market-anchored predictions

`predict.py` now blends its 1X2 and O/U 2.5 probabilities with bookmaker
odds whenever the fixture appears in football-data.co.uk's upcoming-fixtures
feed (downloaded automatically to `data/_fixtures/`, cached 12h). Blend
weights live in `models/blend_weights.json` and were fitted out-of-sample by
[`scripts/blend.py`](scripts/blend.py).

**The model does not beat the bookmaker, and the weights say so.** Re-checked
on 684 matches that are out-of-sample for *both* the per-league ensembles and
the pooled model (Jun–Aug 2026): the de-vigged book scores 1.0032 log-loss,
the production model 1.0334, and fitting the log-pool puts **zero** weight on
the model — giving it weight makes the held-out half worse. So where a fixture
has odds, the numbers you see are essentially the market's, tempered. The
model earns its keep on matches with **no** odds, and in Toto, where the
opponent is the crowd rather than the book. Treat a large "Model − Book" gap
as disagreement worth investigating, not as a value bet.

Per-league weights were tested and did not beat global weights (kept in
`evaluate` for future re-checks). Refit after each retrain:

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
over `data/global/international.csv`, feeding two heads whose 1X2 outputs are
log-pooled (`MIX_MULTINOM = 0.60` on the direct head):

- a **Poisson score grid** with the Dixon-Coles low-score correction, which
  also gives O/U and BTTS;
- a **softmax regression straight to 1X2**, free of the Poisson bottleneck and
  so able to shape the draw probability directly.

The grid is then tilted so its own 1X2 marginals match the blend, keeping O/U
and BTTS coherent with the 1X2 actually reported.

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

`backtest` reports the three-tournament reference (WC 2018, WC 2022, Euro
2024, pre-tournament data only) — around 1.02 log-loss and ~79% top-2. Note
that 179 matches cannot separate two models: the standard error on log-loss
there is ~0.05, larger than any realistic improvement. The blend above was
chosen instead on a **rolling 11,076-match evaluation** (all internationals
from 2015, goal model refit yearly on prior data only):

| model | log-loss | top-2 |
|---|---|---|
| Poisson only | 0.8778 | 82.8% |
| + Dixon-Coles tau | 0.8770 | 83.1% |
| **60/40 blend (shipped)** | **0.8743** | **83.7%** |

Paired *t* = +4.42 against tau alone; significant on qualifiers and
friendlies, neutral on major tournaments, never worse. Top-2 is what a Toto
"double" covers, hence the emphasis. Knockout scores in the dataset include
extra time, which slightly biases evaluation against draws.

`wc2026` seeds the group simulation with results already played and simulates
only the remaining fixtures, sampling scorelines from the corrected grid. Once
the knockouts begin it says so rather than pretending group advancement still
applies.

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
