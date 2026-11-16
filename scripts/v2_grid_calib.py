#!/usr/bin/env python3
"""
Grid calibration: what goes into the score grid before it is tilted to the odds
===============================================================================
The app reads every goal number (over/under, both score, correct score) off
one scoreline grid. This script fits and checks the two corrections that
scripts/v2_goals.serve() applies to that grid:

  mix     model grid ^ w  x  odds grid ^ (1 - w), where the odds grid is the
          Poisson grid whose two goal rates reproduce the 1X2 odds;
  shape   nine multipliers by winning margin and by which draw.

Nothing is retrained. The two bundles stay as they are; the result is one
small file, models/v2/grid_calibration.json. Remove it (or set "enabled" to
false) and the app serves the plain tilt again.

Protocol, as everywhere in v2: the weight was chosen on the 2022-23 windows;
the multipliers used on a test window are fitted on earlier windows only; the
numbers that count are the 2024-26 windows, against the grid served before.

    python -m scripts.v2_grid_calib check            # walk-forward evidence
    python -m scripts.v2_grid_calib check --weights  # ... and the choice of w
    python -m scripts.v2_grid_calib build            # fit to the freeze, write the file
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from scripts import v2_bundle
from scripts import v2_goals as vg
from scripts import v2_train as v2

warnings.filterwarnings('ignore')

CALIB = os.path.join(v2.MODEL_DIR, 'grid_calibration.json')
WEIGHT_MODEL = 0.5          # chosen on the tuning windows: `check --weights`
OOS = os.path.join(v2.RUN_DIR, 'grid_oos')
# what the predictor can be asked for -> (rows it is fitted on, tilt targets, model weight)
MODES = ('model', 'odds', 'odds_ou', 'nomodel', 'nomodel_ou')
# where the evidence is measured: label -> (rows, mode)
SITUATIONS = {
    'leagues with 1X2 and over/under odds': ('rich', 'odds_ou'),
    'leagues with 1X2 odds only': ('sparse', 'odds'),
    'no odds, model only': ('model', 'model'),
    'pair no model knows, 1X2 and over/under odds': ('rich', 'nomodel_ou'),
    'pair no model knows, 1X2 odds only': ('sparse', 'nomodel'),
}


def load_calibration() -> dict | None:
    """The calibration the app serves with, or None (plain tilt)."""
    try:
        with open(CALIB, encoding='utf-8') as f:
            c = json.load(f)
    except (OSError, ValueError):
        return None
    return c if c.get('enabled', True) and c.get('theta') else None


# =============================================================================
# OUT-OF-SAMPLE GRIDS
# =============================================================================
def _ou_price(df, over, under):
    o = pd.to_numeric(df.get(over), errors='coerce').to_numpy(float)
    u = pd.to_numeric(df.get(under), errors='coerce').to_numpy(float)
    ok = np.isfinite(o) & np.isfinite(u) & (o > 1) & (u > 1)
    return np.where(ok, (1 / o) / (1 / o + 1 / np.where(ok, u, 1.0)), np.nan)


def _norm(p):
    return p / p.sum(axis=1, keepdims=True)


def oos(verbose=True) -> dict:
    """Every tuning and test window: the goals model fitted before the window,
    its grids for the window, the prices, and the goal rates read off the
    1X2 price. Cached; rebuilt when the data or the recipe changes."""
    _, metag = v2_bundle.serving_goals()
    recipe = metag['recipe']
    frame = v2_bundle.fresh_cache(recipe['features'], verbose=False)
    key = f"{recipe['name']}|{len(frame)}|{frame.Date.max().date()}|{vg.ODDS_RHO}"
    if os.path.isfile(OOS + '.npz') and os.path.isfile(OOS + '.pkl'):
        z = np.load(OOS + '.npz', allow_pickle=False)
        if str(z['key']) == key:
            return _assemble(pd.read_pickle(OOS + '.pkl'), z['grids'].astype(float), z['rates'])
    t0 = time.time()
    metas, grids = [], []
    for s0 in v2.TUNE_STARTS + v2.TEST_STARTS:
        start = pd.Timestamp(s0)
        test = frame[(frame.Date >= start) & (frame.Date < start + pd.Timedelta(days=v2.TEST_DAYS))]
        if test.empty:
            continue
        grids.append(vg.fit_window(recipe, frame, s0).predict_grid(test).astype(np.float32))
        m = test[['match_id', 'Date', 'league', 'FTHG', 'FTAG', 't_avg_pre_H', 't_avg_pre_D', 't_avg_pre_A',
                  't_avg_close_H', 't_avg_close_D', 't_avg_close_A', 'Avg>2.5', 'Avg<2.5']].copy()
        m['fold'] = s0
        metas.append(m)
        if verbose:
            print(f'    goals model as of {s0}: {len(test):,} matches', flush=True)
    meta, grids = pd.concat(metas, ignore_index=True), np.concatenate(grids)
    pre = meta[['t_avg_pre_H', 't_avg_pre_D', 't_avg_pre_A']].to_numpy(float)
    close = meta[['t_avg_close_H', 't_avg_close_D', 't_avg_close_A']].to_numpy(float)
    price = np.where(np.isfinite(pre).all(axis=1, keepdims=True), pre, close)
    rates = vg.rates_from_1x2(price)
    os.makedirs(v2.RUN_DIR, exist_ok=True)
    meta.to_pickle(OOS + '.pkl')
    np.savez_compressed(OOS + '.npz', key=key, grids=grids, rates=rates)
    if verbose:
        print(f'    {len(meta):,} out-of-sample grids and odds rates in {time.time() - t0:.0f}s')
    return _assemble(meta, grids.astype(float), rates)


def _assemble(meta, grids, rates) -> dict:
    pre = meta[['t_avg_pre_H', 't_avg_pre_D', 't_avg_pre_A']].to_numpy(float)
    close = meta[['t_avg_close_H', 't_avg_close_D', 't_avg_close_A']].to_numpy(float)
    has_pre, has_close = np.isfinite(pre).all(axis=1), np.isfinite(close).all(axis=1)
    ou = _ou_price(meta, 'Avg>2.5', 'Avg<2.5')
    m1, _ = v2_bundle.load_bundle()
    name = m1.recipe['name']
    runs = []
    for tag in ('tune', 'test'):
        p = os.path.join(v2.RUN_DIR, f'{name}__{tag}.pkl')
        if not os.path.isfile(p):
            sys.exit(f'  missing {os.path.relpath(p, v2.ROOT)} — run: python -m scripts.v2_train select')
        runs.append(pd.read_pickle(p))
    r = pd.concat(runs).drop_duplicates('match_id').set_index('match_id')
    model = r.reindex(meta.match_id)[['pH', 'pD', 'pA']].to_numpy(float)
    folds = list(v2.TUNE_STARTS + v2.TEST_STARTS)
    return {
        'meta': meta, 'grids': grids, 'rates': rates,
        'price': _norm(np.where(has_pre[:, None], pre, close)), 'ou': ou, 'model': model,
        'rows': {'rich': has_pre & np.isfinite(ou), 'sparse': ~has_pre & has_close,
                 'priced': has_pre | has_close, 'model': np.isfinite(model).all(axis=1)},
        'fold': meta.fold.map({s: k for k, s in enumerate(folds)}).to_numpy(), 'folds': folds,
        'hg': np.minimum(meta.FTHG.to_numpy(int), vg.G), 'ag': np.minimum(meta.FTAG.to_numpy(int), vg.G),
        'dates': meta.Date.to_numpy(),
    }


# =============================================================================
# FIT
# =============================================================================
def _mode(d: dict, mode: str):
    """(rows the mode is fitted on, 1X2 target, O/U target or None, model weight)."""
    if mode == 'model':
        return d['rows']['model'], d['model'], None, 1.0
    w = 0.0 if mode.startswith('nomodel') else WEIGHT_MODEL
    if mode.endswith('_ou'):
        return d['rows']['rich'], d['price'], d['ou'], w
    return d['rows']['priced'], d['price'], None, w


def _seed(d: dict, idx, w: float):
    if w == 1.0:
        return d['grids'][idx]
    odds = vg.grid(d['rates'][idx, 0], d['rates'][idx, 1], vg.ODDS_RHO)
    if w == 0.0:
        return odds
    s = np.exp(w * np.log(np.clip(d['grids'][idx], 1e-300, None)) + (1 - w) * np.log(np.clip(odds, 1e-300, None)))
    return s / s.sum(axis=(1, 2), keepdims=True)


def fit_mode(d: dict, mode: str, before) -> np.ndarray:
    """Shape multipliers of one mode from the rows selected by `before`."""
    rows, p, ou, w = _mode(d, mode)
    idx = np.where(rows & before)[0]
    return vg.fit_shape(_seed(d, idx, w), p[idx], None if ou is None else ou[idx], d['hg'][idx], d['ag'][idx])


# =============================================================================
# EVIDENCE
# =============================================================================
def _ll_bin(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _losses(g, hg, ag) -> dict:
    tot = vg._I + vg._J
    out = {'score': -np.log(np.clip(g[np.arange(len(hg)), hg, ag], 1e-12, None)),
           'btts': _ll_bin(g[:, (vg._I > 0) & (vg._J > 0)].sum(axis=1), ((hg > 0) & (ag > 0)).astype(int)),
           'top1': (g.reshape(len(hg), -1).argmax(axis=1) == hg * (vg.G + 1) + ag).astype(float)}
    for name, k in (('over 1.5', 2), ('over 2.5', 3), ('over 3.5', 4)):
        out[name] = _ll_bin(g[:, tot >= k].sum(axis=1), (hg + ag >= k).astype(int))
    return out


def _boot(d, dates, B=2000, seed=0):
    wk = pd.to_datetime(pd.Series(np.asarray(dates))).dt.strftime('%G-%V').to_numpy()
    u, inv = np.unique(wk, return_inverse=True)
    s, n = np.bincount(inv, d), np.bincount(inv)
    idx = np.random.default_rng(seed).integers(0, len(u), (B, len(u)))
    m = s[idx].sum(1) / n[idx].sum(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def walk_forward(d: dict, label: str, weight: float | None = None, shape: bool = True,
                 windows=None) -> dict:
    """One situation on the given windows: the grid served before (model grid,
    plain tilt) against the calibrated one, multipliers from earlier windows."""
    rows_key, mode = SITUATIONS[label]
    _, p, ou, w = _mode(d, mode)
    w = w if weight is None else weight
    windows = range(len(v2.TUNE_STARTS), len(d['folds'])) if windows is None else windows
    old, new, dates = [], [], []
    for k in windows:
        te = np.where(d['rows'][rows_key] & (d['fold'] == k))[0]
        if not len(te):
            continue
        o_te = None if ou is None else ou[te]
        seed = _seed(d, te, w)
        if shape:
            cal_rows, cp, cou, _ = _mode(d, mode)
            ci = np.where(cal_rows & (d['fold'] < k))[0]
            theta = vg.fit_shape(_seed(d, ci, w), cp[ci], None if cou is None else cou[ci], d['hg'][ci], d['ag'][ci])
            seed = seed * vg.shape_matrix(theta)[None]
        old.append(_losses(vg.tilt_grid(d['grids'][te], p[te], o_te), d['hg'][te], d['ag'][te]))
        new.append(_losses(vg.tilt_grid(seed, p[te], o_te), d['hg'][te], d['ag'][te]))
        dates.append(d['dates'][te])
    cat = lambda L: {k: np.concatenate([x[k] for x in L]) for k in L[0]}                # noqa: E731
    old, new, dates = cat(old), cat(new), np.concatenate(dates)
    out = {'n': int(len(dates)), 'top1_before': float(old['top1'].mean()), 'top1_after': float(new['top1'].mean())}
    for k in ('score', 'btts', 'over 1.5', 'over 2.5', 'over 3.5'):
        g, lo, hi = _boot(old[k] - new[k], dates)
        out[k] = {'before': float(old[k].mean()), 'after': float(new[k].mean()), 'gain': g, 'ci': [lo, hi]}
    return out


def _print(label, r):
    cells = '  '.join(f"{k} {r[k]['gain']:+.4f} [{r[k]['ci'][0]:+.4f}, {r[k]['ci'][1]:+.4f}]"
                      for k in ('score', 'btts', 'over 2.5', 'over 3.5'))
    print(f"  {label:<46} n={r['n']:>6,}  {cells}  top-1 {r['top1_before']:.1%} -> {r['top1_after']:.1%}")


def check(weights: bool = False) -> dict:
    d = oos()
    n_tune = len(v2.TUNE_STARTS)
    if weights:
        print('\n  The mix weight (model share), without shape multipliers. Chosen on the tuning windows.')
        for label in ('leagues with 1X2 and over/under odds', 'leagues with 1X2 odds only'):
            for tag, wins in (('tuning 2022-23', range(0, n_tune)), ('test 2024-26', range(n_tune, len(d['folds'])))):
                for w in (.75, .5, .25, 0.0):
                    r = walk_forward(d, label, weight=w, shape=False, windows=wins)
                    print(f"    {label:<38} {tag:<15} w={w:.2f}  score {r['score']['gain']:+.5f} "
                          f"[{r['score']['ci'][0]:+.5f}, {r['score']['ci'][1]:+.5f}]  btts {r['btts']['gain']:+.5f}  "
                          f"over 2.5 {r['over 2.5']['gain']:+.5f}")
    print(f'\n  Test windows 2024-26, log-loss gain against the grid served before (positive is better).')
    print(f'  Multipliers for each window are fitted on earlier windows only; model weight {WEIGHT_MODEL}.')
    out = {}
    for label in SITUATIONS:
        out[label] = walk_forward(d, label)
        _print(label, out[label])
    return out


def build() -> str:
    d = oos()
    _, metag = v2_bundle.serving_goals()
    m1, meta1 = v2_bundle.load_bundle()
    freeze = max(pd.Timestamp(metag['trained_through']), pd.Timestamp(meta1['trained_through']))
    before = d['dates'] <= np.datetime64(freeze)
    evidence = check()
    theta, n = {}, {}
    for mode in MODES:
        rows = _mode(d, mode)[0]
        theta[mode] = [round(float(x), 5) for x in fit_mode(d, mode, before)]
        n[mode] = int((rows & before).sum())
    out = {
        'enabled': True,
        'weight_model': WEIGHT_MODEL,
        'rho': vg.ODDS_RHO,
        'groups': list(vg.SHAPE_GROUPS),
        'theta': theta,
        'fitted': {'built_at': pd.Timestamp.now().isoformat(timespec='seconds'),
                   'matches_through': str(freeze.date()), 'n': n,
                   'goals_recipe': metag['recipe']['name'], 'bundle_1x2': meta1['tag'],
                   'bundle_goals': metag['tag'], 'windows': d['folds']},
        'evidence_test_2024_26': evidence,
    }
    with open(CALIB, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f'\n  multipliers (first group of each result = 1), fitted on matches through {freeze.date()}:')
    for mode in MODES:
        print(f"    {mode:<11} n={n[mode]:>6,}  " + '  '.join(
            f'{g}:{np.exp(t):.3f}' for g, t in zip(vg.SHAPE_GROUPS, theta[mode])))
    print(f'  wrote {os.path.relpath(CALIB, v2.ROOT)}')
    return CALIB


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('command', choices=['check', 'build'])
    ap.add_argument('--weights', action='store_true', help='also show how the mix weight was chosen')
    a = ap.parse_args()
    if a.command == 'check':
        check(a.weights)
    else:
        build()


if __name__ == '__main__':
    main()
