#!/usr/bin/env python3
"""
V2 bundles and shadow predictions
=================================
A bundle is a frozen v2 model trained on everything available up to its build
day, with the recipe, data fingerprint and backtest numbers that justify it.
Building one does NOT change what the app serves: bundles live in models/v2/,
and the app keeps loading models/tier1/ until a bundle is promoted by hand.

Shadow mode runs a bundle next to production on real upcoming fixtures and
logs both, so the promotion decision can rest on matches played after the
bundle was frozen — not only on the backtest.

    python -m scripts.v2_bundle build --recipe reports/v2/selected_recipe.json
    python -m scripts.v2_bundle shadow          # log v2 + production for upcoming fixtures
    python -m scripts.v2_bundle grade           # score the shadow log once results are in
    python -m scripts.v2_bundle status
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
import warnings
from collections import defaultdict, deque
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd

from scripts import config, market
from scripts import research_features as rf
from scripts import v2_train as v2

warnings.filterwarnings('ignore')

LATEST = os.path.join(v2.MODEL_DIR, 'LATEST')
SHADOW_LOG = os.path.join(v2.REPORT_DIR, 'shadow', 'shadow_log.csv')
STATE_DIR = os.path.join(v2.REPORT_DIR, 'state')


# =============================================================================
# DATA FRESHNESS
# =============================================================================
def raw_fingerprint() -> str:
    """Same fingerprint build_cache records: sha256 over every raw file's hash."""
    manifest = []
    for league in sorted(config.LEAGUE_REGISTRY):
        for path in sorted(Path(config.DATA_DIR, league).glob('*.csv')):
            manifest.append({'path': str(path.relative_to(config.PROJECT_ROOT)),
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def feature_set(recipe: dict) -> str:
    members = recipe.get('members') or [recipe]
    return 'mkt' if any(m.get('features') == 'mkt' for m in members) else 'v1'


def fresh_cache(features: str = 'v1', verbose=True) -> pd.DataFrame:
    """The feature cache, rebuilt if any raw league file changed since."""
    path = v2.cache_path(features)
    fp = raw_fingerprint()
    if os.path.isfile(path) and os.path.isfile(path + '.json'):
        with open(path + '.json') as f:
            meta = json.load(f)
        if (meta.get('raw_files_sha256') == fp
                and meta.get('feature_version') == rf.version(features == 'mkt')):
            if verbose:
                print(f"  feature cache is current (data through {meta['date_max']})")
            return v2.load_cache(features)
    if verbose:
        print("  raw data or feature code changed since the cache was built — rebuilding")
    return v2.build_cache(verbose=verbose, features=features)


# =============================================================================
# LIVE FEATURE STATE  (rebuilt from the current data, cached by fingerprint)
# =============================================================================
def _freeze(state: rf.FeatureState) -> rf.FeatureState:
    s = copy.copy(state)
    s.league_results = dict(state.league_results)   # lambdas don't pickle
    s.h2h = dict(state.h2h)
    return s


def _thaw(state: rf.FeatureState) -> rf.FeatureState:
    state.league_results = defaultdict(lambda: deque(maxlen=150), state.league_results)
    state.h2h = defaultdict(lambda: deque(maxlen=6), state.h2h)
    return state


def live_state(market: bool = False, verbose=True) -> rf.FeatureState:
    """FeatureState after every completed match in data/, i.e. ready to
    featurize any fixture on a later day."""
    fp = raw_fingerprint()
    version = rf.version(market)
    path = os.path.join(STATE_DIR, f'state_{version}_{fp[:16]}.pkl')
    if os.path.isfile(path):
        return _thaw(joblib.load(path))
    t0 = time.time()
    raw, _ = rf.load_raw()
    _, state = rf.replay(raw, rf.FeatureState(market=market))
    os.makedirs(STATE_DIR, exist_ok=True)
    for old in os.listdir(STATE_DIR):     # keep only current states per version
        if old.startswith(f'state_{version}_'):
            os.remove(os.path.join(STATE_DIR, old))
    joblib.dump(_freeze(state), path)
    if verbose:
        print(f"  replayed {len(raw):,} matches into a live state in "
              f"{time.time()-t0:.0f}s (through {state.last_day.date()})")
    return state


# =============================================================================
# BUILD
# =============================================================================
def _git_head() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                       cwd=config.PROJECT_ROOT, text=True).strip()
    except Exception:
        return ''


def build(recipe: dict, tag: str | None = None) -> str:
    features = feature_set(recipe)
    frame = fresh_cache(features)
    last = frame.Date.max()
    asof = last + pd.Timedelta(days=1)
    tag = tag or f"{pd.Timestamp.now():%Y-%m-%d}_{recipe['name']}"
    out_dir = os.path.join(v2.MODEL_DIR, tag)
    if os.path.exists(out_dir):
        sys.exit(f"  {out_dir} exists — bundles are immutable; pick another --tag")
    t0 = time.time()
    model, info = v2.fit_window(recipe, frame, asof)
    os.makedirs(out_dir)
    joblib.dump(model, os.path.join(out_dir, 'model.joblib'))
    with open(v2.cache_path(features) + '.json') as f:
        cache_meta = json.load(f)
    meta = {
        'tag': tag,
        'built_at': pd.Timestamp.now().isoformat(timespec='seconds'),
        'git_head': _git_head(),
        'recipe': recipe,
        'feature_set': features,
        'feature_version': rf.version(features == 'mkt'),
        'trained_through': str(last.date()),
        'n_fit': info['n_fit'], 'n_cal': info['n_cal'],
        'temperature': model.temperature,
        'data': cache_meta,
        'class_order': ['A', 'D', 'H'],
        'serving': 'shadow only — the app still serves models/tier1',
    }
    for name in ('selection.json', f"compare_{recipe['name']}.json"):
        p = os.path.join(v2.REPORT_DIR, name)
        if os.path.isfile(p):
            with open(p) as f:
                meta[name.removesuffix('.json')] = json.load(f)
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    with open(LATEST, 'w') as f:
        f.write(tag + '\n')
    print(f"  bundle {tag}: trained through {last.date()} "
          f"(fit {info['n_fit']:,}, cal {info['n_cal']:,}, T={model.temperature:.3f}) "
          f"in {time.time()-t0:.0f}s -> {os.path.relpath(out_dir, config.PROJECT_ROOT)}")
    return out_dir


def load_bundle(tag: str | None = None):
    if tag is None:
        if not os.path.isfile(LATEST):
            sys.exit("  no v2 bundle yet — run: python -m scripts.v2_bundle build")
        with open(LATEST) as f:
            tag = f.read().strip()
    d = os.path.join(v2.MODEL_DIR, tag)
    with open(os.path.join(d, 'meta.json')) as f:
        meta = json.load(f)
    expected = rf.version(meta.get('feature_set', 'v1') == 'mkt')
    if meta['feature_version'] != expected:
        sys.exit(f"  bundle {tag} was built for features {meta['feature_version']}, "
                 f"code is {expected} — rebuild it")
    return joblib.load(os.path.join(d, 'model.joblib')), meta


# =============================================================================
# PREDICT
# =============================================================================
def featurize(fixtures: pd.DataFrame, state: rf.FeatureState) -> pd.DataFrame:
    """Causal features for upcoming fixtures (Date, league, HomeTeam, AwayTeam).
    Fixtures on or before the state's last completed day are dropped: their
    day is already part of history, so they cannot be predicted causally."""
    ok = fixtures.Date > state.last_day
    rows = [dict(state.fixture(r), **{k: r[k] for k in rf.KEY})
            for r in fixtures[ok].to_dict('records')]
    return pd.DataFrame(rows)


def predict(fixtures: pd.DataFrame, tag: str | None = None,
            state: rf.FeatureState | None = None) -> pd.DataFrame:
    model, meta = load_bundle(tag)
    state = state or live_state(market=meta.get('feature_set') == 'mkt')
    feats = featurize(fixtures, state)
    if feats.empty:
        return feats
    feats[['v2A', 'v2D', 'v2H']] = model.predict_proba(feats)
    feats['bundle'] = meta['tag']
    return feats


# =============================================================================
# SHADOW LOG
# =============================================================================
def _production_1x2(fx: pd.DataFrame) -> pd.DataFrame:
    """Production's model-only 1X2 (before the market anchor) per fixture."""
    from scripts import data_loader, utils
    from scripts import predict as predict_mod
    hist = data_loader.load_processed_data()
    team_stats = utils.get_team_stats_table(hist)
    team_to_league = utils.get_team_to_league_map(hist)
    out = []
    for r in fx.to_dict('records'):
        try:
            p = predict_mod.predict_match(r['HomeTeam'], r['AwayTeam'], team_stats,
                                          team_to_league, hist, include_xg=False,
                                          prediction_date=r['Date'])
            q = p.get('1x2_model', p.get('1x2'))
            out.append((q['away'], q['draw'], q['home']))
        except Exception:
            out.append((np.nan, np.nan, np.nan))
    return pd.DataFrame(out, columns=['prodA', 'prodD', 'prodH'], index=fx.index)


def shadow(days: int = 7):
    from scripts import fixtures as fixtures_mod
    fixtures_mod.fetch()
    fx = fixtures_mod.load(fetch_if_missing=False)
    now = pd.Timestamp.now().normalize()
    fx = fx[(fx.Date >= now) & (fx.Date <= now + pd.Timedelta(days=days))]
    fx = fx.dropna(subset=['league']).drop_duplicates(rf.KEY).reset_index(drop=True)
    if fx.empty:
        print("  no upcoming fixtures in the feed")
        return
    pred = predict(fx)
    if pred.empty:
        print("  no fixtures after the last completed day in the data")
        return
    pred = pred.merge(fx, on=rf.KEY, how='left')
    mk = market.shin(pred[['OddsA', 'OddsD', 'OddsH']].to_numpy(float))
    pred[['mktA', 'mktD', 'mktH']] = mk
    pred = pd.concat([pred, _production_1x2(pred)], axis=1)
    pred['logged_at'] = pd.Timestamp.now().isoformat(timespec='seconds')
    cols = rf.KEY + ['bundle', 'logged_at', 'v2A', 'v2D', 'v2H',
                     'prodA', 'prodD', 'prodH', 'mktA', 'mktD', 'mktH',
                     'OddsH', 'OddsD', 'OddsA']
    new = pred[cols]
    os.makedirs(os.path.dirname(SHADOW_LOG), exist_ok=True)
    if os.path.isfile(SHADOW_LOG):
        old = pd.read_csv(SHADOW_LOG, parse_dates=['Date'])
        both = pd.concat([old, new], ignore_index=True)
    else:
        both = new
    # keep the latest pre-match prediction per fixture and bundle
    both = both.sort_values('logged_at').drop_duplicates(rf.KEY + ['bundle'], keep='last')
    both.to_csv(SHADOW_LOG, index=False)
    print(f"  logged {len(new)} fixtures ({new.league.nunique()} leagues) -> "
          f"{os.path.relpath(SHADOW_LOG, config.PROJECT_ROOT)} "
          f"({len(both)} total)")
    show = new.assign(**{c: new[c].round(2) for c in ['v2H', 'v2D', 'v2A',
                                                        'prodH', 'prodD', 'prodA',
                                                        'mktH', 'mktD', 'mktA']})
    print(show[['Date', 'league', 'HomeTeam', 'AwayTeam', 'v2H', 'v2D', 'v2A',
                'prodH', 'prodD', 'prodA', 'mktH', 'mktD', 'mktA']]
          .head(40).to_string(index=False))


def grade():
    if not os.path.isfile(SHADOW_LOG):
        sys.exit("  no shadow log yet — run: python -m scripts.v2_bundle shadow")
    log = pd.read_csv(SHADOW_LOG, parse_dates=['Date'])
    raw, _ = rf.load_raw()
    res = raw[rf.KEY + ['FTR']]
    j = log.merge(res, on=rf.KEY, how='inner')
    if j.empty:
        print(f"  {len(log)} logged fixtures, none played yet")
        return
    y = j.FTR.map({'A': 0, 'D': 1, 'H': 2}).to_numpy(int)
    print(f"  graded {len(j)} of {len(log)} logged fixtures "
          f"({j.Date.min().date()} .. {j.Date.max().date()})")
    for label, cols in (('v2', ['v2A', 'v2D', 'v2H']),
                        ('production model', ['prodA', 'prodD', 'prodH']),
                        ('market', ['mktA', 'mktD', 'mktH'])):
        P = j[cols].to_numpy(float)
        ok = np.isfinite(P).all(axis=1)
        if not ok.any():
            continue
        ll = -np.log(np.clip(P[ok][np.arange(ok.sum()), y[ok]], 1e-12, None))
        print(f"  {label:<18} n={ok.sum():>5}  log-loss {ll.mean():.4f}  "
              f"accuracy {(P[ok].argmax(1) == y[ok]).mean():.3f}")
    both = j[['v2A', 'prodA']].notna().all(axis=1).to_numpy()
    if both.sum() >= 30:
        Pv = j.loc[both, ['v2A', 'v2D', 'v2H']].to_numpy(float)
        Pp = j.loc[both, ['prodA', 'prodD', 'prodH']].to_numpy(float)
        yy = y[both]
        d = (-np.log(Pp[np.arange(len(yy)), yy]) + np.log(Pv[np.arange(len(yy)), yy]))
        wk = pd.Series(d).groupby(j.loc[both, 'Date'].dt.strftime('%G-%V').to_numpy()).agg(['sum', 'count'])
        idx = np.random.default_rng(0).integers(0, len(wk), size=(2000, len(wk)))
        boots = wk['sum'].to_numpy()[idx].sum(1) / wk['count'].to_numpy()[idx].sum(1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"  v2 gain over production (paired, {both.sum()} matches, {len(wk)} weeks): "
              f"{d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")


def status():
    if not os.path.isfile(LATEST):
        print("  no v2 bundle yet")
    else:
        _, meta = load_bundle()
        print(f"  latest bundle {meta['tag']} — trained through {meta['trained_through']}, "
              f"built {meta['built_at']}, {meta['serving']}")
    tier1 = config.get_tier_dir(tier=1)
    print(f"  production still loads {os.path.relpath(tier1, config.PROJECT_ROOT)}")
    if os.path.isfile(SHADOW_LOG):
        print(f"  shadow log: {len(pd.read_csv(SHADOW_LOG))} fixtures")


def main():
    ap = argparse.ArgumentParser(description='V2 bundles and shadow predictions')
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--recipe', default=os.path.join(v2.REPORT_DIR, 'selected_recipe.json'))
    b.add_argument('--tag')
    s = sub.add_parser('shadow')
    s.add_argument('--days', type=int, default=7)
    sub.add_parser('grade')
    sub.add_parser('status')
    a = ap.parse_args()
    if a.cmd == 'build':
        with open(a.recipe) as f:
            build(json.load(f), a.tag)
    elif a.cmd == 'shadow':
        shadow(a.days)
    elif a.cmd == 'grade':
        grade()
    else:
        status()


if __name__ == '__main__':
    main()
