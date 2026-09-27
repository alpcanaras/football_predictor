#!/usr/bin/env python3
"""Isolated chronological research training and comparison; never production.

python -m scripts.research_train evaluate --leagues british_pl german turkish
python -m scripts.research_train train --cutoff 2026-09-27 --leagues british_pl german turkish

All output directories are new, immutable run IDs under reports/research/.
Historical price returns are explicitly hypothetical (quote times unknown).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import uuid

import joblib
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize_scalar
from scipy.special import softmax
import sklearn
from sklearn.metrics import log_loss
import xgboost as xgb

from scripts import config
from scripts.research_features import VERSION, FeatureState, feature_columns, load_raw, replay

ROOT = Path(config.PROJECT_ROOT)
DEFAULT_LEAGUES = ['british_pl', 'german', 'spanish', 'italian', 'turkish']
Q = ['qA', 'qD', 'qH']
STRATEGIES = ['market', 'market_calibrated', 'football', 'residual']


def labels(df):
    return df.FTR.map({'A': 0, 'D': 1, 'H': 2}).to_numpy(int)


def matrix(df):
    # The declared registry is a schema, not an observed future-team encoding.
    x = df[feature_columns()].astype(float).reset_index(drop=True)
    for league in sorted(config.LEAGUE_REGISTRY):
        x['league_'+league] = (df.league.to_numpy() == league).astype(float)
    return x


def market_mask(df):
    return np.isfinite(df[Q]).all(axis=1) & df[Q].gt(0).all(axis=1)


def weights(dates, half_life):
    if half_life <= 0:
        return np.ones(len(dates))
    w = .5 ** ((dates.max() - dates).dt.days.to_numpy(float) / half_life)
    return w / w.mean()


def scale(p, temperature):
    return softmax(np.log(np.clip(p, 1e-12, 1)) / temperature, axis=1)


def correct(q, p, strength):
    return softmax((1-strength)*np.log(np.clip(q, 1e-12, 1))
                   + strength*np.log(np.clip(p, 1e-12, 1)), axis=1)


def fit_scalar(fn, bounds):
    result = minimize_scalar(fn, bounds=bounds, method='bounded')
    if not result.success:
        raise RuntimeError(f'Calibration failed: {result.message}')
    candidates = [bounds[0], bounds[1], float(result.x)]
    return min(candidates, key=fn)


def temporal_split(frame, cutoff, calibration_days=90, lookback_days=1460):
    cutoff = pd.Timestamp(cutoff).normalize()
    if calibration_days < 1 or lookback_days <= calibration_days:
        raise ValueError('Require 0 < calibration_days < lookback_days')
    cal_start = cutoff - pd.Timedelta(days=calibration_days)
    lower = cutoff - pd.Timedelta(days=lookback_days)
    fit = frame[(frame.Date >= lower) & (frame.Date < cal_start)].copy()
    cal = frame[(frame.Date >= cal_start) & (frame.Date < cutoff)].copy()
    if fit.empty or cal.empty:
        raise ValueError('Empty fit or calibration period')
    assert fit.Date.max() < cal.Date.min() and cal.Date.max() < cutoff
    return fit, cal


def fit_bundle(frame, cutoff, args):
    fit, cal = temporal_split(frame, cutoff, args.calibration_days, args.lookback_days)
    fit_market, cal_market = fit[market_mask(fit)], cal[market_mask(cal)]
    for name, data in [('fit', fit), ('fit_market', fit_market),
                       ('calibration', cal), ('calibration_market', cal_market)]:
        minimum = 100 if name.startswith('fit') else 60
        if len(data) < minimum or set(labels(data)) != {0, 1, 2}:
            raise ValueError(f'{name}: need >= {minimum} matches and all 3 outcomes')
    params = dict(n_estimators=args.trees, max_depth=3, learning_rate=.035,
                  min_child_weight=30, reg_lambda=10, reg_alpha=.1,
                  subsample=.85, colsample_bytree=.85, n_jobs=4,
                  objective='multi:softprob', num_class=3, tree_method='hist',
                  random_state=42)
    football = xgb.XGBClassifier(**params)
    football.fit(matrix(fit), labels(fit), sample_weight=weights(fit.Date, args.half_life))
    pc = football.predict_proba(matrix(cal))
    temp = fit_scalar(lambda t: log_loss(labels(cal), scale(pc, t), labels=[0, 1, 2]), (.5, 3.0))
    residual = xgb.XGBClassifier(**params)
    residual.fit(matrix(fit_market), labels(fit_market),
                 base_margin=np.log(fit_market[Q].to_numpy()),
                 sample_weight=weights(fit_market.Date, args.half_life))
    qc = cal_market[Q].to_numpy()
    rc = residual.predict_proba(matrix(cal_market), base_margin=np.log(qc))
    strength = fit_scalar(lambda a: log_loss(labels(cal_market), correct(qc, rc, a), labels=[0, 1, 2]), (0., 1.))
    book_temp = fit_scalar(lambda t: log_loss(labels(cal_market), scale(qc, t), labels=[0, 1, 2]), (.5, 3.))
    return dict(football=football, residual=residual, temperature=temp,
                residual_strength=strength, book_temperature=book_temp,
                cutoff=str(pd.Timestamp(cutoff).date()), feature_version=VERSION,
                feature_columns=list(matrix(fit).columns),
                leagues=sorted(frame.league.unique()),
                fit_start=str(fit.Date.min().date()), fit_end=str(fit.Date.max().date()),
                calibration_start=str(cal.Date.min().date()), calibration_end=str(cal.Date.max().date()),
                n_fit=len(fit), n_fit_market=len(fit_market), n_cal=len(cal),
                n_cal_market=len(cal_market), params=params,
                elo_k=args.elo_k, season_retention=args.season_retention)


def predict_bundle(bundle, frame, enforce_cutoff=True):
    if bundle['feature_version'] != VERSION:
        raise ValueError('Feature version mismatch')
    if enforce_cutoff and (frame.Date < pd.Timestamp(bundle['cutoff'])).any():
        raise ValueError('Refusing predictions before the training/calibration cutoff')
    if set(frame.league) - set(bundle['leagues']):
        raise ValueError('Unknown training league')
    x = matrix(frame)
    if list(x.columns) != bundle['feature_columns']:
        raise ValueError('Feature schema mismatch')
    p = scale(bundle['football'].predict_proba(x), bundle['temperature'])
    output = {'football': p}
    q = frame[Q].to_numpy(float)
    good = market_mask(frame).to_numpy()
    for strategy in ['market', 'market_calibrated', 'residual']:
        output[strategy] = np.full((len(frame), 3), np.nan)
    if good.any():
        output['market'][good] = q[good]
        output['market_calibrated'][good] = scale(q[good], bundle['book_temperature'])
        residual = bundle['residual'].predict_proba(x.loc[good], base_margin=np.log(q[good]))
        output['residual'][good] = correct(q[good], residual, bundle['residual_strength'])
    return output


def paired_interval(df, challenger, repetitions=1000):
    y = df.y.to_numpy(int)
    b = df[[f'market_{c}' for c in 'ADH']].to_numpy()
    p = df[[f'{challenger}_{c}' for c in 'ADH']].to_numpy()
    loss_difference = np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)) - np.log(b[np.arange(len(y)), y])
    # Weeks are the resampling units; all bets/matches in a week travel together.
    dates = pd.to_datetime(df.Date).dt.to_period('W').astype(str)
    group = pd.DataFrame({'week': dates.to_numpy(), 'd': loss_difference}).groupby('week').d.agg(['sum', 'count'])
    if len(group) < 8:
        return {'improvement': float(loss_difference.mean()), 'ci95': None, 'weeks': len(group)}
    rng = np.random.default_rng(42)
    ix = rng.integers(len(group), size=(repetitions, len(group)))
    delta = group['sum'].to_numpy()[ix].sum(axis=1) / group['count'].to_numpy()[ix].sum(axis=1)
    return {'improvement': float(loss_difference.mean()),
            'ci95': np.quantile(delta, [.025, .975]).tolist(), 'weeks': len(group)}


def metrics(df):
    if df.empty:
        return {'n': 0}
    result = {'n': len(df), 'strategies': {}}
    y = df.y.to_numpy(int)
    odds = df[['betA', 'betD', 'betH']].to_numpy(float)
    for name in STRATEGIES:
        p = df[[f'{name}_{c}' for c in 'ADH']].to_numpy()
        pick = p.argmax(axis=1)
        ev = p * odds - 1
        valid = np.isfinite(odds).all(axis=1) & (odds > 1).all(axis=1)
        ev[~np.isfinite(ev)] = -np.inf
        choice = ev.argmax(axis=1)
        selected = valid & (ev[np.arange(len(y)), choice] >= .05)
        profit = np.where(choice == y, odds[np.arange(len(y)), choice] - 1, -1)
        result['strategies'][name] = {
            'log_loss': float(log_loss(y, p, labels=[0, 1, 2])),
            'brier_multiclass': float(np.square(p-np.eye(3)[y]).sum(axis=1).mean()),
            'accuracy': float((pick == y).mean()),
            'hypothetical_bets_5pct': int(selected.sum()),
            'hypothetical_roi': float(profit[selected].mean()) if selected.any() else None,
        }
        if name != 'market':
            result['strategies'][name]['paired_vs_market'] = paired_interval(df, name)
    return result


def git_revision():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False, default=str)+'\n')


def new_run(args, source_files):
    run = ROOT / 'reports' / 'research' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    source_code = {name: hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()
                   for name in ['research_features.py', 'research_train.py']}
    write_json(run/'manifest.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        arguments=vars(args), feature_version=VERSION, git_commit=git_revision(), code_sha256=source_code,
        environment=dict(python=platform.python_version(), pandas=pd.__version__,
                         numpy=np.__version__, xgboost=xgb.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__),
        source_files=source_files,
        limitations=['Historical odds have unknown quote timestamps; ROI is hypothetical.',
                     'Date-level replay assumes previous-day results available; no publication timestamps.',
                     'All inspected periods are research data; prospective evaluation is still required.',
                     'Promoted teams start from a league prior; no transfer or lineup information.']))
    return run


def evaluate(frame, args, run):
    cutoffs = sorted(pd.Timestamp(s).normalize() for s in args.cutoffs)
    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError('Duplicate evaluation cutoffs')
    for a, b in zip(cutoffs, cutoffs[1:]):
        if a + pd.Timedelta(days=args.test_days) > b:
            raise ValueError('Evaluation windows must not overlap')
    predictions, folds = [], []
    for cutoff in cutoffs:
        end = cutoff + pd.Timedelta(days=args.test_days)
        test = frame[(frame.Date >= cutoff) & (frame.Date < end) & market_mask(frame)].copy()
        if test.empty:
            raise ValueError(f'No comparable matches for {cutoff.date()}')
        print(f'Fold {cutoff.date()}: evaluating {len(test)} matches', flush=True)
        bundle = fit_bundle(frame, cutoff, args)
        probs = predict_bundle(bundle, test)
        out = test[['match_id', 'Date', 'league', 'HomeTeam', 'AwayTeam',
                    'HomeSeasonN', 'AwaySeasonN', 'odds_source', 'betA', 'betD', 'betH']].copy()
        out['y'], out['fold'] = labels(test), str(cutoff.date())
        for name, p in probs.items():
            out[[f'{name}_{c}' for c in 'ADH']] = p
        predictions.append(out)
        folds.append({'cutoff': str(cutoff.date()), 'test_end_exclusive': str(end.date()),
                      'training': {k: v for k, v in bundle.items() if k not in ['football', 'residual']},
                      'metrics': metrics(out)})
        joblib.dump(bundle, run/f'bundle-{cutoff.date()}.joblib')
        print({k: round(v['log_loss'], 4) for k, v in folds[-1]['metrics']['strategies'].items()}, flush=True)
    out = pd.concat(predictions, ignore_index=True)
    if out.match_id.duplicated().any():
        raise ValueError('Duplicate test matches across folds')
    out.to_csv(run/'predictions.csv', index=False)
    football_pick = out[[f'football_{c}' for c in 'ADH']].to_numpy().argmax(axis=1)
    book_pick = out[[f'market_{c}' for c in 'ADH']].to_numpy().argmax(axis=1)
    summary = dict(folds=folds, overall=metrics(out),
        early_season=metrics(out[np.minimum(out.HomeSeasonN, out.AwaySeasonN) < 6]),
        top_pick_disagreements=metrics(out[football_pick != book_pick]),
        by_league={league: metrics(group) for league, group in out.groupby('league')},
        warning='Exploratory results, not proof of tradable profit. Do not select a league or threshold on these results and call it a test.')
    write_json(run/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['evaluate', 'train'])
    parser.add_argument('--leagues', nargs='+', default=DEFAULT_LEAGUES)
    parser.add_argument('--cutoffs', nargs='+', default=['2024-08-01', '2025-08-01', '2026-08-01'])
    parser.add_argument('--test-days', type=int, default=60)
    parser.add_argument('--cutoff', default=str(pd.Timestamp.now().date()))
    parser.add_argument('--calibration-days', type=int, default=90)
    parser.add_argument('--lookback-days', type=int, default=1460)
    parser.add_argument('--half-life', type=float, default=500.)
    parser.add_argument('--trees', type=int, default=250)
    parser.add_argument('--elo-k', type=float, default=20.)
    parser.add_argument('--season-retention', type=float, default=.85)
    args = parser.parse_args()
    if args.trees < 1 or args.test_days < 1:
        parser.error('trees and test-days must be positive')
    raw, sources = load_raw(args.leagues)
    frame, _ = replay(raw, FeatureState(args.elo_k, args.season_retention))
    run = new_run(args, sources)
    print(f'Run: {run}\nReplayed {len(frame):,} matches', flush=True)
    if args.command == 'evaluate':
        result = evaluate(frame, args, run)
        print(json.dumps(result['overall'], indent=2))
    else:
        bundle = fit_bundle(frame, args.cutoff, args)
        joblib.dump(bundle, run/'bundle.joblib')
        write_json(run/'training.json', {k: v for k, v in bundle.items() if k not in ['football', 'residual']})
        print('Saved isolated challenger bundle; production models untouched.')
    print(f'Results: {run}', flush=True)


if __name__ == '__main__':
    main()
