"""
Pooled Cross-League Model (partial pooling)
===========================================
ONE model per market trained on all leagues at once, with the league added
as an explicit set of one-hot features. This is the statistically-correct
middle ground between:

  - per-league models  (your current setup: 28 small, noisy models)
  - a naive pool       (ignores the league -> blurs goal-rich vs goal-poor)

Adding one column per league lets a tree model isolate any league that
genuinely behaves differently (scoring level, home advantage, draw rate),
while SHARING the universal structure (how an Elo edge, form, or rest maps
to a result) across all ~75k matches. Goal-rich leagues stay goal-rich;
small leagues borrow strength from the rest.

Only the CORE features are used (they exist for every league); the rich
shot/corner features stay in the per-league models. Nothing here overwrites
your production models — pooled models live in a separate 'pooled' model set,
and `evaluate` reports pooled vs per-league vs book per league so you can
decide, with numbers, whether to adopt it.

    # 1. (heavy, needs green light) train the pooled models
    python scripts/pooled.py train --markets 1x2 ou25 btts

    # 2. (cheap) compare pooled vs per-league vs book on the OOS window
    python scripts/pooled.py evaluate
"""

import argparse
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from scripts import config
from scripts import data_loader
from scripts import utils

warnings.filterwarnings('ignore')

POOLED_SET = 'pooled'
POOLED_KEY = 'POOLED'                 # stands in for "league" in model filenames
LEAGUE_PREFIX = 'lg_'

OBJECTIVES = {
    '1x2': 'multi:softprob',
    'ou25': 'binary:logistic',
    'ou15': 'binary:logistic',
    'ou35': 'binary:logistic',
    'btts': 'binary:logistic',
}
TARGETS = {
    '1x2': 'result_label', 'ou25': 'over_2_5', 'ou15': 'over_1_5',
    'ou35': 'over_3_5', 'btts': 'btts',
}


def _meta_path() -> str:
    return os.path.join(config.get_tier_dir(model_set=POOLED_SET),
                        'pooled_meta.json')


# =============================================================================
# FEATURE MATRIX  (core features + one-hot league)
# =============================================================================
def build_pooled_X(df: pd.DataFrame, columns: list | None = None):
    """Return (X, columns): core features plus one dummy column per league.

    If `columns` is given (the training-time order), the result is reindexed
    to exactly those columns so prediction/eval matrices always line up,
    with unseen leagues collapsing to all-zero dummies.
    """
    core = config.FEATURES_CORE
    X = df[core].copy()
    dummies = pd.get_dummies(df['league'].astype(str), prefix=LEAGUE_PREFIX.rstrip('_'))
    X = pd.concat([X.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X, list(X.columns)


# =============================================================================
# TRAINING  (gated behind the `train` subcommand; reuses train.py primitives)
# =============================================================================
def train(markets, n_trials, cutoff, decay, verbose=True):
    from scripts import train as trainer

    all_data = data_loader.load_processed_data()
    if cutoff is not None:
        cutoff_ts = pd.Timestamp(cutoff)
        train_data = all_data[(all_data['Date'] < cutoff_ts)
                              & (all_data['Date'].dt.year >= config.TRAIN_START_YEAR)]
    else:
        train_data = all_data[all_data['Date'].dt.year >= config.TRAIN_START_YEAR]
    train_data = train_data.sort_values('Date').reset_index(drop=True)

    X_full, columns = build_pooled_X(train_data)

    print('=' * 64)
    print(f"  POOLED TRAIN  rows={len(train_data)}  leagues="
          f"{train_data['league'].nunique()}  features={len(columns)}")
    print(f"  engines={config.MODEL_TYPES}  trials/model={n_trials}  "
          f"cutoff={cutoff}")
    print('=' * 64)

    for market in markets:
        target = TARGETS[market]
        objective = OBJECTIVES[market]
        mask = train_data[target].notna() & X_full.notna().all(axis=1)
        Xm = X_full[mask.values]
        ym = train_data.loc[mask.values, target].astype(int)
        sw = trainer._time_decay_weights(train_data.loc[mask.values, 'Date'], decay)

        t0 = time.time()
        print(f"\n  [{market}] {len(Xm)} rows, objective={objective} ...")
        models = trainer._train_cls(Xm, ym, objective, sw, n_trials=n_trials)
        for engine, d in models.items():
            utils.save_model(d['model'], market, POOLED_KEY,
                             model_set=POOLED_SET, engine=engine)
            print(f"    {engine}: saved  cv_ll={d['cv_score']:.4f}  "
                  f"iters={d['n_iters']}  {'cal' if d['calibrated'] else 'no-cal'}")
        print(f"  [{market}] done in {time.time() - t0:.0f}s")

    os.makedirs(os.path.dirname(_meta_path()), exist_ok=True)
    with open(_meta_path(), 'w', encoding='utf-8') as f:
        json.dump({'columns': columns, 'markets': list(markets),
                   'cutoff': cutoff, 'n_train': int(len(train_data))}, f, indent=2)
    print(f"\n  Saved pooled model set + meta -> {_meta_path()}")


# =============================================================================
# PREDICTION
# =============================================================================
def pooled_proba(market, df_rows):
    """Engine-averaged pooled predict_proba for the given rows (any leagues).

    Returns None if the pooled model set hasn't been trained yet.
    """
    if not os.path.exists(_meta_path()):
        return None
    with open(_meta_path(), encoding='utf-8') as f:
        columns = json.load(f)['columns']
    X, _ = build_pooled_X(df_rows, columns=columns)

    probas = []
    for eng in config.MODEL_TYPES:
        m = utils.load_model(market, POOLED_KEY, model_set=POOLED_SET, engine=eng)
        if m is not None:
            try:
                probas.append(m.predict_proba(X))
            except Exception:
                pass
    if not probas:
        return None
    return np.mean(probas, axis=0)


# =============================================================================
# EVALUATION  (cheap; the number that decides whether pooling is worth it)
# =============================================================================
def evaluate(cutoff, market='1x2'):
    from scripts import blend

    if not os.path.exists(_meta_path()):
        print("No pooled models yet. Run:  python scripts/pooled.py train")
        return

    target = TARGETS[market]
    n_classes = 3 if market == '1x2' else 2
    labels = list(range(n_classes))

    all_data = data_loader.load_processed_data()
    cutoff_ts = pd.Timestamp(cutoff)
    leagues = [l for l in utils.get_available_leagues()
               if l not in blend.IN_SAMPLE_LEAGUES]

    print(f"\n  POOLED vs PER-LEAGUE vs BOOK  —  {market.upper()}  "
          f"(OOS from {cutoff})\n")
    print(f"  {'League':<20} {'N':>5} {'Pooled':>8} {'PerLeague':>10} "
          f"{'Book':>8}   winner")
    print('  ' + '-' * 64)

    agg = {'pooled': [], 'per': [], 'book': [], 'y': []}
    odds_cols = ['OddsH', 'OddsD', 'OddsA'] if market == '1x2' \
        else ['OddsOver25', 'OddsUnder25']

    for league in sorted(leagues):
        feats = config.get_features_for_league(league)
        ldf = all_data[all_data['league'] == league]
        test = ldf[ldf['Date'] >= cutoff_ts].dropna(
            subset=feats + [target] + odds_cols)
        if len(test) < 20:
            continue

        p_pool = pooled_proba(market, test)
        p_per = blend.ensemble_proba(league, test[feats]) if market == '1x2' \
            else blend.ensemble_proba_market(market, league, test[feats])
        if p_pool is None or p_per is None:
            continue

        if market == '1x2':
            p_book = blend.implied_proba(test)
        else:
            inv_o = 1.0 / test['OddsOver25'].to_numpy(float)
            inv_u = 1.0 / test['OddsUnder25'].to_numpy(float)
            over = inv_o / (inv_o + inv_u)
            p_book = np.column_stack([1 - over, over])

        y = test[target].to_numpy(int)
        ll_pool = log_loss(y, p_pool, labels=labels)
        ll_per = log_loss(y, p_per, labels=labels)
        ll_book = log_loss(y, p_book, labels=labels)

        best = min([('Pooled', ll_pool), ('PerLeague', ll_per),
                    ('Book', ll_book)], key=lambda x: x[1])[0]
        print(f"  {league:<20} {len(test):>5} {ll_pool:>8.4f} "
              f"{ll_per:>10.4f} {ll_book:>8.4f}   {best}")

        agg['pooled'].append((ll_pool, len(test)))
        agg['per'].append((ll_per, len(test)))
        agg['book'].append((ll_book, len(test)))
        agg['y'].append(len(test))

    def wmean(pairs):
        s = sum(ll * n for ll, n in pairs)
        return s / sum(n for _, n in pairs)

    if agg['pooled']:
        print('  ' + '-' * 64)
        print(f"  {'WEIGHTED OVERALL':<20} {sum(agg['y']):>5} "
              f"{wmean(agg['pooled']):>8.4f} {wmean(agg['per']):>10.4f} "
              f"{wmean(agg['book']):>8.4f}")
        print("\n  (Lower log-loss = better. 'PerLeague' is your current "
              "production model.)")


def main():
    parser = argparse.ArgumentParser(description='Pooled cross-league model')
    sub = parser.add_subparsers(dest='command', required=True)

    p_tr = sub.add_parser('train', help='Train pooled models (HEAVY)')
    p_tr.add_argument('--markets', nargs='+',
                      default=['1x2', 'ou25', 'btts'],
                      choices=list(TARGETS))
    p_tr.add_argument('--n-trials', type=int, default=config.OPTUNA_N_TRIALS)
    p_tr.add_argument('--cutoff', default=None,
                      help='Hold out matches on/after this date (for evaluate)')
    p_tr.add_argument('--decay-half-life-days', type=float,
                      default=config.TIME_DECAY_HALF_LIFE_DAYS)

    p_ev = sub.add_parser('evaluate', help='Pooled vs per-league vs book')
    p_ev.add_argument('--cutoff', default='2026-02-01')
    p_ev.add_argument('--market', default='1x2', choices=list(TARGETS))

    args = parser.parse_args()
    if args.command == 'train':
        train(args.markets, args.n_trials, args.cutoff,
              args.decay_half_life_days)
    else:
        evaluate(args.cutoff, args.market)


if __name__ == '__main__':
    main()
