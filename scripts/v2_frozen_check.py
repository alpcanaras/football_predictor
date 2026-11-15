#!/usr/bin/env python3
"""
Frozen-bundle check: the served models on matches played after they were frozen
===============================================================================
The cleanest test there is. Both bundles were fixed before any of these
matches kicked off, and the features come from the causal replay (as of each
match's day), exactly as in live use. Nothing is fitted here.

It scores, on the same matches:
    1X2       the frozen ensemble against the early price (what the live feed
              carries) and the closing price
    goals     over/under 2.5 against the over/under price; both-teams-score;
              the scoreline grid (plain, and tilted to the price as the app
              serves it) against league score frequencies

    python -m scripts.v2_frozen_check                  # since the freeze
    python -m scripts.v2_frozen_check --since 2026-10-15
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from scripts import v2_bundle
from scripts import v2_goals as vg
from scripts import v2_train as v2

warnings.filterwarnings('ignore')
HDA = ['H', 'D', 'A']


def _ll(p, y):
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))


def _ll_bin(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot(d, dates, B=2000, seed=0):
    """Mean of d with a 95% weekly block-bootstrap interval."""
    wk = pd.to_datetime(pd.Series(np.asarray(dates))).dt.strftime('%G-%V').to_numpy()
    u, inv = np.unique(wk, return_inverse=True)
    s, n = np.bincount(inv, d), np.bincount(inv)
    idx = np.random.default_rng(seed).integers(0, len(u), (B, len(u)))
    m = s[idx].sum(1) / n[idx].sum(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), len(u)


def _cols(frame, stem):
    p = frame[[f'{stem}_{c}' for c in HDA]].to_numpy(float)
    return p / p.sum(axis=1, keepdims=True)


def _ou_price(frame, over, under):
    o = pd.to_numeric(frame.get(over), errors='coerce').to_numpy(float)
    u = pd.to_numeric(frame.get(under), errors='coerce').to_numpy(float)
    ok = np.isfinite(o) & np.isfinite(u) & (o > 1) & (u > 1)
    return np.where(ok, (1 / o) / (1 / o + 1 / np.where(ok, u, 1.0)), np.nan)


def _versus(name, mine, other, dates, out, key):
    """Print loss of `mine` against `other` on the rows where both exist."""
    ok = np.isfinite(mine) & np.isfinite(other)
    if ok.sum() < 50:
        print(f'    {name:<34} too few matches ({int(ok.sum())})')
        return
    d, lo, hi, wk = boot(other[ok] - mine[ok], np.asarray(dates)[ok])
    print(f'    {name:<34} model {mine[ok].mean():.5f} | price {other[ok].mean():.5f} | '
          f'model gain {d:+.5f} [{lo:+.5f}, {hi:+.5f}]  ({int(ok.sum()):,} matches, {wk} weeks)')
    out[key] = {'n': int(ok.sum()), 'weeks': wk, 'model': float(mine[ok].mean()),
                'price': float(other[ok].mean()), 'gain': d, 'ci': [lo, hi]}


def run(since: str | None = None) -> dict:
    m1, meta1 = v2_bundle.load_bundle()
    mg, metag = v2_bundle.serving_goals()
    frozen = max(pd.Timestamp(meta1['trained_through']), pd.Timestamp(metag['trained_through']))
    start = pd.Timestamp(since) if since else frozen + pd.Timedelta(days=1)
    if start <= frozen:
        sys.exit(f'  --since must be after the freeze ({frozen.date()}): earlier matches were trained on')
    frame = v2_bundle.fresh_cache('mkttot', verbose=False)
    past, test = frame[frame.Date < start], frame[frame.Date >= start].reset_index(drop=True)
    if len(test) < 100:
        sys.exit(f'  only {len(test)} matches since {start.date()} — update the data first')
    y = test.FTR.map({'H': 0, 'D': 1, 'A': 2}).to_numpy(int)
    out = {'since': str(start.date()), 'through': str(test.Date.max().date()), 'n': int(len(test)),
           'bundle_1x2': meta1['tag'], 'bundle_goals': metag['tag']}
    print(f"  1X2 bundle {meta1['tag']}, goals bundle {metag['tag']} (frozen {frozen.date()})")
    print(f"  {len(test):,} matches, {test.league.nunique()} leagues, "
          f"{start.date()} .. {test.Date.max().date()}")

    # ---- 1X2
    p = m1.predict_proba(test)[:, ::-1]                    # [A, D, H] -> [H, D, A]
    l_model = _ll(p, y)
    print(f'\n  1X2 (log-loss, lower is better; gain > 0 means the model beats the price)')
    print(f'    {"model, all matches":<34} {l_model.mean():.5f}   accuracy {(p.argmax(1) == y).mean():.3f}')
    out['1x2'] = {'model_all': float(l_model.mean())}
    for label, stem in (('against the early price', 't_avg_pre'), ('against the closing price', 't_avg_close')):
        q = _cols(test, stem)
        lq = np.where(np.isfinite(q).all(axis=1), _ll(np.nan_to_num(q, nan=1 / 3), y), np.nan)
        _versus(label, l_model, lq, test.Date, out['1x2'], stem)

    # ---- goals
    grid = mg.predict_grid(test)
    mk = vg.grid_markets(grid)
    yo = ((test.FTHG + test.FTAG) >= 3).to_numpy(int)
    yb = ((test.FTHG > 0) & (test.FTAG > 0)).to_numpy(int)
    out['goals'] = {}
    print(f'\n  over/under 2.5')
    l_ou = _ll_bin(mk['ou25'], yo)
    for label, cols in (('against the early O/U price', ('Avg>2.5', 'Avg<2.5')),
                        ('against the closing O/U price', ('AvgC>2.5', 'AvgC<2.5'))):
        q = _ou_price(test, *cols)
        _versus(label, l_ou, np.where(np.isfinite(q), _ll_bin(np.nan_to_num(q, nan=.5), yo), np.nan),
                test.Date, out['goals'], 'ou_' + cols[0])
    base_b = np.full(len(test), ((past.FTHG > 0) & (past.FTAG > 0)).tail(20000).mean())
    d, lo, hi, _ = boot(_ll_bin(base_b, yb) - _ll_bin(mk['btts'], yb), test.Date)
    print(f'\n  both teams score: model {_ll_bin(mk["btts"], yb).mean():.5f} | flat base rate '
          f'{_ll_bin(base_b, yb).mean():.5f} | model gain {d:+.5f} [{lo:+.5f}, {hi:+.5f}]')
    out['goals']['btts'] = {'model': float(_ll_bin(mk['btts'], yb).mean()), 'gain_vs_base': d, 'ci': [lo, hi]}

    # ---- scoreline grid, as served: tilted to the price where there is one
    hi_, ai_ = vg._actual_cells(test)
    cell = lambda g: -np.log(np.clip(g[np.arange(len(test)), hi_, ai_], 1e-12, None))
    prior = vg.league_prior(past[past.Date >= start - pd.Timedelta(days=1460)], test)
    pre = _cols(test, 't_avg_pre')
    close = _cols(test, 't_avg_close')
    price = np.where(np.isfinite(pre).all(axis=1, keepdims=True), pre, close)
    has = np.isfinite(price).all(axis=1)
    ou_pre, ou_close = _ou_price(test, 'Avg>2.5', 'Avg<2.5'), _ou_price(test, 'AvgC>2.5', 'AvgC<2.5')
    ou = np.where(np.isfinite(ou_pre), ou_pre, ou_close)
    ou = np.where(np.isfinite(ou), ou, mk['ou25'])
    served = grid.copy()
    served[has] = vg.tilt_grid(grid[has], price[has], ou[has])
    print(f'\n  correct score (log-loss over the scoreline; top-1 = most likely score was right)')
    base = cell(prior)
    out['score'] = {}
    for label, g in (('league score frequencies', prior), ('model grid', grid),
                     ('grid tilted to the price (served)', served)):
        l = cell(g)
        top = (g.reshape(len(test), -1).argmax(1) == hi_ * (vg.G + 1) + ai_).mean()
        d, lo, hi, _ = boot(base - l, test.Date)
        print(f'    {label:<34} {l.mean():.4f}   top-1 {top:.3f}   gain vs frequencies {d:+.4f} [{lo:+.4f}, {hi:+.4f}]')
        out['score'][label] = {'ll': float(l.mean()), 'top1': float(top), 'gain': d, 'ci': [lo, hi]}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--since', help='first match day to score (default: the day after the freeze)')
    a = ap.parse_args()
    out = run(a.since)
    path = os.path.join(v2.REPORT_DIR, 'frozen_check.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n  saved {os.path.relpath(path, v2.ROOT)}')


if __name__ == '__main__':
    main()
