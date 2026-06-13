#!/usr/bin/env python3
"""
Walk-forward 1X2 "top-2" coverage (honest holdout).

For each test match, take the two outcomes with highest predicted probability
(Home / Draw / Away). Count how often the actual FTR is one of those two.

Uses the same date split as train.validate_models: train on Date < cutoff,
evaluate on the last VALIDATION_DAYS.

Example:
  python scripts/validate_top2_1x2.py
  python scripts/validate_top2_1x2.py --n-trials 30
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from scripts import config
from scripts import data_loader
from scripts.train import (
    MARKET_DEFS,
    _train_cls,
    _time_decay_weights,
    N_TRIALS as DEFAULT_N_TRIALS,
)

# result_label: 0=A, 1=D, 2=H (see features.py)
_LBL_TO_FTR = {0: 'A', 1: 'D', 2: 'H'}


def _avg_predict_proba(cls_models, X):
    probas, classes = [], None
    for eng in config.MODEL_TYPES:
        if eng not in cls_models:
            continue
        m = cls_models[eng]['model']
        p = m.predict_proba(X)
        cl = np.asarray(m.classes_)
        if classes is None:
            classes = cl
            probas.append(p)
        else:
            if np.array_equal(cl, classes):
                probas.append(p)
            else:
                aligned = np.zeros_like(p)
                for j, c in enumerate(classes):
                    idx = np.where(cl == c)[0]
                    if len(idx):
                        aligned[:, j] = p[:, idx[0]]
                probas.append(aligned)
    return np.mean(probas, axis=0), classes


def top2_hits(proba, classes, ftr_series):
    """Boolean array: true if actual FTR is in the two most likely classes."""
    hits = []
    n_cls = len(classes)
    for i in range(len(proba)):
        p = proba[i]
        order = np.argsort(-p)
        if n_cls >= 2:
            c1, c2 = int(classes[order[0]]), int(classes[order[1]])
            set_ftr = {_LBL_TO_FTR[c1], _LBL_TO_FTR[c2]}
        else:
            set_ftr = {_LBL_TO_FTR[int(c)] for c in classes}
        hits.append(ftr_series.iloc[i] in set_ftr)
    return np.asarray(hits, dtype=bool)


def main():
    parser = argparse.ArgumentParser(description='1X2 top-2 outcome coverage (walk-forward)')
    parser.add_argument(
        '--n-trials',
        type=int,
        default=max(DEFAULT_N_TRIALS // 4, 20),
        help='Optuna trials per engine (pipeline Phase 3 used OPTUNA_N_TRIALS//4)',
    )
    parser.add_argument(
        '--validation-days',
        type=int,
        default=int(getattr(config, 'VALIDATION_DAYS', 60)),
    )
    args = parser.parse_args()

    print('Loading processed data...')
    all_data = data_loader.load_processed_data()
    base = all_data[all_data['Date'].dt.year >= config.TRAIN_START_YEAR]
    max_date = base['Date'].max()
    cutoff = max_date - pd.Timedelta(days=args.validation_days)
    train_d = base[base['Date'] < cutoff]
    test_d = base[base['Date'] >= cutoff]

    print(f'  Train: {len(train_d):,} rows (before {cutoff.date()})')
    print(f'  Test:  {len(test_d):,} rows ({cutoff.date()} .. {max_date.date()})')
    print(f'  Optuna trials per engine: {args.n_trials}\n')

    target, objective = MARKET_DEFS['1x2']
    leagues = sorted(l for l in base['league'].unique() if l in config.LEAGUE_REGISTRY)

    all_hits = []

    for league in leagues:
        info = config.LEAGUE_REGISTRY.get(league, {})
        disp = info.get('display_name', league)
        lt = train_d[train_d['league'] == league]
        le = test_d[test_d['league'] == league]

        if len(lt) < config.MIN_MATCHES_PER_LEAGUE or len(le) < 5:
            continue

        feats = config.get_features_for_league(league)
        tr = lt.dropna(subset=feats + [target])
        te = le.dropna(subset=feats + [target, 'FTR'])
        if len(tr) < 50 or len(te) < 5:
            continue

        sw = _time_decay_weights(tr['Date'], getattr(config, 'TIME_DECAY_HALF_LIFE_DAYS', None))
        cls_models = _train_cls(tr[feats], tr[target], objective, sw, n_trials=args.n_trials)
        avg_p, classes = _avg_predict_proba(cls_models, te[feats])
        hits = top2_hits(avg_p, classes, te['FTR'].reset_index(drop=True))

        n = len(hits)
        k = int(hits.sum())

        all_hits.append(hits)

        print(f'  {disp:<28}  top-2: {k}/{n}  ({100*k/n:.1f}%)')

    if not all_hits:
        print('No league results.')
        return

    h = np.concatenate(all_hits)
    print(f"\n{'='*60}")
    print('  ALL LEAGUES POOLED (honest walk-forward test window)')
    print(f"{'='*60}")
    print(f"  Matches where actual result was in the model's top-2 outcomes:")
    print(f"    {int(h.sum())} / {len(h)}  ({100 * h.mean():.2f}%)")
    print()
    print('  Baseline: picking any 2 of 3 at random covers 2/3 ~= 66.7% of')
    print('  outcomes in expectation; “always include Home+Draw” is not a fair')
    print('  baseline — the model chooses which pair per match.')
    print('=' * 60)


if __name__ == '__main__':
    main()
