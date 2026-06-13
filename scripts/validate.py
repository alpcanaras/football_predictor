#!/usr/bin/env python3
"""
Model Validation Script
=======================
Out-of-sample validation on the temporal test set, with optional
model-vs-bookmaker comparison.

Usage:
    python scripts/validate.py                    # Validate all
    python scripts/validate.py --compare          # + bookmaker comparison
    python scripts/validate.py --summary          # Data summary only
"""

import argparse
import warnings
import sys
import os
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error

from scripts import config
from scripts import data_loader
from scripts import utils

warnings.filterwarnings('ignore')


# =============================================================================
# HELPERS
# =============================================================================
def _get_test_data(all_data: pd.DataFrame) -> pd.DataFrame:
    cutoff = pd.Timestamp(config.TRAIN_CUTOFF)
    return all_data[(all_data['Date'] >= cutoff) &
                    (all_data['Date'].dt.year >= config.TRAIN_START_YEAR)].copy()


# =============================================================================
# CLASSIFICATION VALIDATION (1X2, O/U 2.5, BTTS)
# =============================================================================
def validate_classification(test_data, model_type, target_col,
                            model_set: str = 'current'):
    labels = {'1x2': '1X2', 'ou25': 'OVER/UNDER 2.5', 'ou15': 'OVER/UNDER 1.5',
              'ou35': 'OVER/UNDER 3.5', 'btts': 'BTTS',
              'ht1x2': 'HALF-TIME 1X2', 'htou05': 'HT OVER/UNDER 0.5'}
    is_multi = model_type in ('1x2', 'ht1x2')

    print(f"\n{'='*72}")
    print(f"  {labels[model_type]} MODEL VALIDATION")
    print(f"{'='*72}")

    for tier in [1, 2]:
        tl = "Tier 1 (no odds)" if tier == 1 else "Tier 2 (with odds)"
        print(f"\n  --- {tl} ---")
        hdr = f"  {'League':<25} {'Acc':>8} {'LogLoss':>9} {'N':>6}"
        if is_multi:
            hdr += f" {'Top2':>8}"
        print(hdr)
        print(f"  {'-' * (len(hdr) - 2)}")

        leagues = utils.get_available_leagues(tier=tier, model_set=model_set)
        tot_correct, tot_n = 0, 0

        for league in leagues:
            model = utils.load_model(model_type, league, tier=tier,
                                     model_set=model_set)
            if not model:
                continue

            feats = config.get_features_for_league(league, tier=tier, market=model_type)
            lt = test_data[test_data['league'] == league].dropna(
                subset=feats + [target_col])
            if len(lt) < 5:
                continue

            X, y = lt[feats], lt[target_col]
            try:
                proba = model.predict_proba(X)
                pred = model.predict(X)
            except Exception:
                continue

            acc = accuracy_score(y, pred)
            try:
                ll = log_loss(y, proba)
            except Exception:
                ll = float('nan')

            line = f"  {league:<25} {acc:>8.1%} {ll:>9.3f} {len(y):>6}"

            if is_multi:
                top2 = np.argsort(proba, axis=1)[:, ::-1][:, :2]
                t2_acc = sum(y.values[i] in top2[i] for i in range(len(y))) / len(y)
                line += f" {t2_acc:>8.1%}"

            print(line)
            tot_correct += int(acc * len(y))
            tot_n += len(y)

        if tot_n:
            print(f"  {'OVERALL':<25} {tot_correct / tot_n:>8.1%} {'':>9} {tot_n:>6}")


# =============================================================================
# xG VALIDATION
# =============================================================================
def validate_xg(test_data, model_set: str = 'current'):
    print(f"\n{'='*72}")
    print(f"  EXPECTED GOALS VALIDATION")
    print(f"{'='*72}")

    for tier in [1, 2]:
        tl = "Tier 1 (no odds)" if tier == 1 else "Tier 2 (with odds)"
        print(f"\n  --- {tl} ---")
        print(f"  {'League':<25} {'MAE-H':>8} {'MAE-A':>8} {'N':>6}")
        print(f"  {'-'*53}")

        td = config.get_tier_dir(tier=tier, model_set=model_set)
        if not os.path.isdir(td):
            continue
        files = glob.glob(os.path.join(td, 'model_xGH_*.joblib'))
        leagues = sorted(
            os.path.basename(f).replace('model_xGH_', '').replace('.joblib', '')
            for f in files)

        for league in leagues:
            mh = utils.load_model('xGH', league, tier=tier,
                                  model_set=model_set)
            ma = utils.load_model('xGA', league, tier=tier,
                                  model_set=model_set)
            if not mh or not ma:
                continue

            feats = config.get_features_for_league(league, tier=tier, market='xg')
            lt = test_data[test_data['league'] == league].dropna(
                subset=feats + ['FTHG', 'FTAG'])
            if len(lt) < 5:
                continue

            X = lt[feats]
            mae_h = mean_absolute_error(lt['FTHG'], mh.predict(X))
            mae_a = mean_absolute_error(lt['FTAG'], ma.predict(X))
            print(f"  {league:<25} {mae_h:>8.3f} {mae_a:>8.3f} {len(lt):>6}")


# =============================================================================
# MODEL vs BOOKMAKER
# =============================================================================
def compare_vs_bookmaker(test_data, model_set: str = 'current'):
    print(f"\n{'='*72}")
    print(f"  MODEL vs BOOKMAKER (1X2)")
    print(f"{'='*72}")
    print(f"\n  {'League':<25} {'Mod Acc':>8} {'Book Acc':>9} "
          f"{'Mod LL':>8} {'Book LL':>9} {'N':>6}")
    print(f"  {'-'*71}")

    # Derive implied probabilities from raw odds if the processed file
    # doesn't carry them.
    if 'ImpliedProbH' not in test_data.columns:
        test_data = test_data.copy()
        inv_h = 1.0 / test_data['OddsH']
        inv_d = 1.0 / test_data['OddsD']
        inv_a = 1.0 / test_data['OddsA']
        overround = inv_h + inv_d + inv_a
        test_data['ImpliedProbH'] = inv_h / overround
        test_data['ImpliedProbD'] = inv_d / overround
        test_data['ImpliedProbA'] = inv_a / overround

    leagues = utils.get_available_leagues(tier=1, model_set=model_set)
    tot = {'m_correct': 0, 'b_correct': 0, 'n': 0}

    for league in leagues:
        model = utils.load_model('1x2', league, tier=1, model_set=model_set)
        if not model:
            continue

        feats = config.get_features_for_league(league, tier=1, market='1x2')
        odds_cols = ['ImpliedProbH', 'ImpliedProbD', 'ImpliedProbA']
        lt = test_data[test_data['league'] == league].dropna(
            subset=feats + ['result_label'] + odds_cols)
        if len(lt) < 10:
            continue

        X, y = lt[feats], lt['result_label']

        # Model
        m_proba = model.predict_proba(X)
        m_pred = model.predict(X)
        m_acc = accuracy_score(y, m_pred)
        try:
            m_ll = log_loss(y, m_proba)
        except Exception:
            m_ll = float('nan')

        # Bookmaker (class order: 0=A, 1=D, 2=H)
        b_proba = lt[['ImpliedProbA', 'ImpliedProbD', 'ImpliedProbH']].values
        b_pred = np.argmax(b_proba, axis=1)
        b_acc = accuracy_score(y, b_pred)
        try:
            b_ll = log_loss(y, b_proba)
        except Exception:
            b_ll = float('nan')

        print(f"  {league:<25} {m_acc:>8.1%} {b_acc:>9.1%} "
              f"{m_ll:>8.3f} {b_ll:>9.3f} {len(y):>6}")

        tot['m_correct'] += int(m_acc * len(y))
        tot['b_correct'] += int(b_acc * len(y))
        tot['n'] += len(y)

    if tot['n']:
        print(f"  {'OVERALL':<25} "
              f"{tot['m_correct']/tot['n']:>8.1%} "
              f"{tot['b_correct']/tot['n']:>9.1%} "
              f"{'':>8} {'':>9} {tot['n']:>6}")


# =============================================================================
# DATA SUMMARY
# =============================================================================
def show_summary(all_data):
    print(f"\n{'='*60}")
    print(f"  DATA SUMMARY")
    print(f"{'='*60}")
    print(f"\n  Total matches : {len(all_data)}")
    print(f"  Date range    : {all_data['Date'].min().date()} to "
          f"{all_data['Date'].max().date()}")
    print(f"  Leagues       : {all_data['league'].nunique()}")
    print(f"  Teams         : {len(utils.get_all_teams(all_data))}")
    print(f"\n  Matches per league:")
    for league, count in all_data['league'].value_counts().items():
        disp = config.LEAGUE_REGISTRY.get(league, {}).get('display_name', league)
        print(f"    {disp:<35} {count:>6}")


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Validate football prediction models')
    parser.add_argument('--models', nargs='+',
                        choices=['1x2', 'ou25', 'ou15', 'ou35', 'btts',
                                 'xg', 'ht1x2', 'htou05', 'all'],
                        default=['all'])
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--compare', action='store_true',
                        help='Compare model vs bookmaker implied probs')
    parser.add_argument('--model-set', type=str, default='current',
                        help="Model set name to validate (default: 'current')")
    args = parser.parse_args()

    print("=" * 60)
    print("FOOTBALL PREDICTOR - MODEL VALIDATION")
    print("=" * 60)
    print(f"Model set: {args.model_set}")

    print("\nLoading data...")
    try:
        all_data = data_loader.load_processed_data()
    except FileNotFoundError:
        print("ERROR: Run 'python scripts/train.py' first.")
        sys.exit(1)

    print(f"Loaded {len(all_data)} matches")

    if args.summary:
        show_summary(all_data)
        return

    test_data = _get_test_data(all_data)
    print(f"Test set: {len(test_data)} matches (from {config.TRAIN_CUTOFF})")

    models = args.models
    if 'all' in models:
        models = ['1x2', 'ou25', 'ou15', 'ou35', 'btts',
                  'xg', 'ht1x2', 'htou05']

    market_targets = {
        '1x2': 'result_label', 'ou25': 'over_2_5', 'ou15': 'over_1_5',
        'ou35': 'over_3_5', 'btts': 'btts', 'ht1x2': 'ht_result',
        'htou05': 'ht_over_0_5',
    }
    for mkt in models:
        if mkt == 'xg':
            validate_xg(test_data, model_set=args.model_set)
        elif mkt in market_targets:
            validate_classification(
                test_data, mkt, market_targets[mkt],
                model_set=args.model_set
            )

    if args.compare:
        compare_vs_bookmaker(test_data, model_set=args.model_set)

    print(f"\n{'='*60}")
    print("VALIDATION COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
