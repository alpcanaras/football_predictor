#!/usr/bin/env python3
"""
V2 vs production, head to head on matches neither has seen
==========================================================
The v2 backtest (scripts/v2_train.py) compares recipes with each other. This
answers the question that decides promotion: is a v2 model better than the
models the app loads today, on matches that are out-of-sample for BOTH?

Production's 1X2 model is two models averaged 50/50, and they were trained at
different times, so there are two honest pools:

  per-league   the per-league XGB+LGBM ensemble alone, scored on matches after
               each league's model file was written (most: 2026-03-24);
  full         per-league 50/50 with the pooled model, exactly as predict.py
               serves it — the pooled model was retrained on everything on
               2026-08-22, so only matches after that are unseen by it.

v2 is trained "as of" the start of each pool (fit + 90-day calibration window
strictly before it), so it never has more information than production had,
and for leagues production retrained later it has less.

The market (Shin-de-vigged pre-match average odds) is scored on the same
matches: when odds exist it is what the app actually plays.

    python -m scripts.v2_compare                       # distilled baseline
    python -m scripts.v2_compare --recipe reports/v2/selected_recipe.json
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

from scripts import config, data_loader, utils
from scripts import blend as blend_mod
from scripts import pooled as pooled_mod
from scripts import v2_train as v2

warnings.filterwarnings('ignore')
KEY = ['league', 'Date', 'HomeTeam', 'AwayTeam']


def _file_day(path: str) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(os.path.getmtime(path), unit='s').normalize()
    except OSError:
        return None


def league_cutoff(league: str) -> pd.Timestamp | None:
    """Day the league's production 1X2 models were last written. Anything
    played on a later day cannot have been in their training data."""
    days = [_file_day(utils._model_path('1x2', league, 1, 'current', eng))
            for eng in config.MODEL_TYPES]
    days = [d for d in days if d is not None]
    return max(days) if days else None


def pooled_cutoff() -> pd.Timestamp | None:
    return _file_day(pooled_mod._meta_path())


def production_pool() -> pd.DataFrame:
    """Production 1X2 probabilities on every match after its league's cutoff."""
    data = data_loader.load_processed_data()
    data['Date'] = pd.to_datetime(data['Date']).dt.normalize()
    p_cut = pooled_cutoff()
    rows = []
    for league in utils.get_available_leagues():
        cut = league_cutoff(league)
        if cut is None:
            continue
        feats = config.get_features_for_league(league)
        ldf = data[(data.league == league) & (data.Date > cut)]
        ldf = ldf.dropna(subset=feats + ['result_label'])
        if ldf.empty:
            continue
        X = ldf[feats]
        pl = blend_mod.ensemble_proba(league, X)
        if pl is None:
            continue
        pl = np.asarray(pl)
        pf, full_ok = pl.copy(), np.zeros(len(ldf), bool)
        if pooled_mod.knows_league(league):
            pp = pooled_mod.pooled_proba('1x2', X.assign(league=league))
            if pp is not None:
                pf = .5 * pl + .5 * np.asarray(pp)
                full_ok = (ldf.Date > p_cut).to_numpy() if p_cut is not None \
                    else np.zeros(len(ldf), bool)
        else:
            # not pooled: the served model is the per-league one alone
            full_ok = np.ones(len(ldf), bool)
        out = ldf[KEY].copy()
        out['cutoff'] = cut
        out[['plA', 'plD', 'plH']] = pl
        out[['pfA', 'pfD', 'pfH']] = pf
        out['full_oos'] = full_ok
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def v2_asof(recipe: dict, asof: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Train v2 as of a date and predict every later match."""
    res = v2.run({**recipe, 'name': f"{recipe['name']}__asof_{asof}"}, frame,
                 [asof], save=False, verbose=False, test_days=3650)
    t = frame.set_index('match_id')
    mk = market_probs(t.loc[res.match_id])
    res[['mA', 'mD', 'mH']] = mk
    res = res.merge(frame[['match_id', 'HomeTeam', 'AwayTeam']], on='match_id')
    res['Date'] = pd.to_datetime(res['Date']).dt.normalize()
    return res


def market_probs(df: pd.DataFrame) -> np.ndarray:
    """Shin pre-match average odds; Bet365 where the average is missing."""
    out = df[['t_avg_pre_A', 't_avg_pre_D', 't_avg_pre_H']].to_numpy(float)
    alt = df[['t_b365_pre_A', 't_b365_pre_D', 't_b365_pre_H']].to_numpy(float)
    miss = ~np.isfinite(out).all(axis=1)
    out[miss] = alt[miss]
    return out


def _ll(P, y):
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None))


def _paired(d: np.ndarray, dates: pd.Series, B=2000, seed=0):
    wk = pd.Series(d).groupby(dates.dt.strftime('%G-%V').to_numpy()).agg(['sum', 'count'])
    s, c = wk['sum'].to_numpy(), wk['count'].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(s), size=(B, len(s)))
    boots = s[idx].sum(1) / c[idx].sum(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return d.mean(), lo, hi, len(s)


def _top2(P, y):
    return (np.argsort(P, axis=1)[:, 1:] == y[:, None]).any(axis=1).mean()


def report(j: pd.DataFrame, prod_cols, label: str) -> dict:
    y = j['y'].to_numpy()
    P_prod = j[prod_cols].to_numpy(float)
    P_v2 = j[['pA', 'pD', 'pH']].to_numpy(float)
    P_mk = j[['mA', 'mD', 'mH']].to_numpy(float)
    has_mk = np.isfinite(P_mk).all(axis=1)
    l_prod, l_v2 = _ll(P_prod, y), _ll(P_v2, y)
    diff, lo, hi, wk = _paired(l_prod - l_v2, j['Date'])
    print(f"\n  {label}: {len(j):,} matches, {j.league.nunique()} leagues, "
          f"{j.Date.min().date()} .. {j.Date.max().date()} ({wk} weeks)")
    print(f"  {'':<24} {'log-loss':>9} {'accuracy':>9} {'top-2':>7}")
    for name, P, l in (('production', P_prod, l_prod), ('v2', P_v2, l_v2)):
        print(f"  {name:<24} {l.mean():>9.5f} {(P.argmax(1) == y).mean():>9.4f} "
              f"{_top2(P, y):>7.4f}")
    print(f"  v2 gain over production: {diff:+.5f}  95% weekly CI "
          f"[{lo:+.5f}, {hi:+.5f}]{'  *' if lo > 0 else ''}")
    out = {'n': int(len(j)), 'weeks': int(wk), 'll_prod': float(l_prod.mean()),
           'll_v2': float(l_v2.mean()), 'gain': float(diff),
           'ci': [float(lo), float(hi)]}
    if has_mk.sum() >= 100:
        m = has_mk
        l_mk = _ll(P_mk[m], y[m])
        sub = j[m]
        d2 = _paired(l_mk - l_v2[m], sub['Date'])
        d3 = _paired(l_mk - l_prod[m], sub['Date'])
        print(f"  with market odds ({m.sum():,}): market {l_mk.mean():.5f} | "
              f"v2 {l_v2[m].mean():.5f} | production {l_prod[m].mean():.5f}")
        print(f"    v2 vs market         {d2[0]:+.5f} [{d2[1]:+.5f}, {d2[2]:+.5f}]")
        print(f"    production vs market {d3[0]:+.5f} [{d3[1]:+.5f}, {d3[2]:+.5f}]")
        out.update(n_market=int(m.sum()), ll_market=float(l_mk.mean()),
                   v2_vs_market=[float(x) for x in d2[:3]],
                   prod_vs_market=[float(x) for x in d3[:3]])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--recipe', help='JSON recipe (default: distilled baseline)')
    args = ap.parse_args()
    if args.recipe:
        with open(args.recipe) as f:
            recipe = json.load(f)
    else:
        recipe = v2.variant(name='distilled_baseline', **v2.BEST_DISTILL)
    print(f"  recipe: {recipe['name']} ({recipe['engine']})")

    prod = production_pool()
    frame = v2.load_cache()
    results = {'recipe': recipe, 'data_max': str(frame.Date.max().date())}

    # per-league pool: v2 as of the earliest production cutoff in it
    asof = prod.cutoff.min() + pd.Timedelta(days=1)
    pred = v2_asof(recipe, str(asof.date()), frame)
    j = prod.merge(pred, on=KEY, how='inner')
    results['per_league'] = report(j, ['plA', 'plD', 'plH'],
                                   'Per-league production models')

    # full served model: only matches after the pooled retrain
    pc = pooled_cutoff()
    if pc is not None:
        asof2 = pc + pd.Timedelta(days=1)
        pred2 = v2_asof(recipe, str(asof2.date()), frame)
        j2 = prod[prod.full_oos].merge(pred2, on=KEY, how='inner')
        if len(j2) >= 200:
            results['full'] = report(j2, ['pfA', 'pfD', 'pfH'],
                                     'Full production model as served (per-league + pooled)')

    os.makedirs(v2.REPORT_DIR, exist_ok=True)
    path = os.path.join(v2.REPORT_DIR, f"compare_{recipe['name']}.json")
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  gain > 0 means v2 is better; * = interval excludes zero")
    print(f"  saved {os.path.relpath(path, config.PROJECT_ROOT)}")


if __name__ == '__main__':
    main()
