#!/usr/bin/env python3
"""
Coupon-level backtest
=====================
Every number the Toto tab shows — P(>=12), the tier ladder, the system
optimiser's choices — rests on two claims:

  1. the per-match probabilities are calibrated, and
  2. matches are independent, so the count of correct picks is Poisson-binomial.

Match-level calibration is easy to check. This checks the pair of them at the
level you actually play: build many synthetic coupons out of real
out-of-sample matches, compare the predicted P(>=threshold) with how often the
coupons really got there.

Sampling matters. Coupons drawn from *the same matchday* carry whatever
correlation real matchdays have (a weekend of upsets hits every row at once);
coupons drawn *across the whole window* are close to independent by
construction. If the same-day hit rate sits above the prediction while the
spread-out one matches, the independence assumption is conservative — good
news for a high threshold, since correlation fattens the tails.

    python scripts/coupon_backtest.py
    python scripts/coupon_backtest.py --since 2026-03-25 --size 13 --threshold 10
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from scripts import config, data_loader, utils
from scripts import blend as blend_mod
from scripts import toto


def build_pool(since: str, use_odds: bool) -> pd.DataFrame:
    """Out-of-sample matches with a model probability and the actual result."""
    all_data = data_loader.load_processed_data()
    rows = []
    for league in utils.get_available_leagues():
        feats = config.get_features_for_league(league)
        ldf = all_data[all_data['league'] == league]
        need = feats + ['result_label']
        test = ldf[ldf['Date'] >= pd.Timestamp(since)].dropna(subset=need)
        if len(test) < 5:
            continue
        p = blend_mod.ensemble_proba(league, test[feats])
        if p is None:
            continue
        p = np.asarray(p)
        if use_odds:
            book = blend_mod.implied_proba(test)
            ok = ~np.isnan(book).any(axis=1)
            p = np.where(ok[:, None], book, p)      # market where available
        rows.append(pd.DataFrame({
            'Date': test['Date'].to_numpy(),
            'league': league,
            'y': test['result_label'].to_numpy(int),
            'pA': p[:, 0], 'pD': p[:, 1], 'pH': p[:, 2],
        }))
    if not rows:
        return pd.DataFrame()
    pool = pd.concat(rows, ignore_index=True)
    P = pool[['pA', 'pD', 'pH']].to_numpy()
    pool['conf'] = P.max(axis=1)
    pool['hit'] = (P.argmax(axis=1) == pool['y'].to_numpy()).astype(int)
    return pool


def sample_coupons(pool: pd.DataFrame, size: int, n: int, same_day: bool,
                   rng) -> tuple[np.ndarray, np.ndarray]:
    """Return (predicted P(>=k) per coupon, realised correct count per coupon)."""
    preds, hits = [], []
    if same_day:
        by_day = [g for _, g in pool.groupby(pool['Date'].dt.date)
                  if len(g) >= size]
        if not by_day:
            return np.array([]), np.array([])
    for _ in range(n):
        if same_day:
            g = by_day[rng.integers(len(by_day))]
            idx = rng.choice(len(g), size=size, replace=False)
            sub = g.iloc[idx]
        else:
            idx = rng.choice(len(pool), size=size, replace=False)
            sub = pool.iloc[idx]
        preds.append(sub['conf'].to_numpy())
        hits.append(int(sub['hit'].sum()))
    return np.array(preds), np.array(hits)


def report(label, qs, hits, threshold, pool_n=None):
    """Print one sampling scheme's results.

    The confidence interval is on the *matches*, not the coupons: coupons are
    resampled from the same pool, so treating each as independent evidence
    would overstate precision enormously.
    """
    if not len(hits):
        print(f"  {label:<28} (not enough matches per day)")
        return
    predicted = np.array([toto.prob_at_least(q, threshold) for q in qs])
    exp_correct = qs.sum(axis=1)
    ci = ''
    if pool_n:
        per_match = hits.mean() / qs.shape[1]
        se = np.sqrt(max(per_match * (1 - per_match), 1e-9) / pool_n)
        ci = f"±{1.96 * se * qs.shape[1]:.2f}"
    print(f"  {label:<28} {len(hits):>6} {exp_correct.mean():>9.2f} "
          f"{hits.mean():>9.2f} {predicted.mean():>11.2%} "
          f"{(hits >= threshold).mean():>9.2%} {hits.std(ddof=1):>7.2f} {ci:>8}")


def main():
    ap = argparse.ArgumentParser(description='Coupon-level backtest')
    ap.add_argument('--since', default='2026-03-25',
                    help='start of the out-of-sample window')
    ap.add_argument('--size', type=int, default=13, help='matches per coupon')
    ap.add_argument('--threshold', type=int, default=10, help='prize threshold')
    ap.add_argument('--n', type=int, default=4000, help='coupons to simulate')
    ap.add_argument('--no-odds', action='store_true',
                    help='model only; by default the market is used where known')
    args = ap.parse_args()

    pool = build_pool(args.since, use_odds=not args.no_odds)
    if pool.empty:
        sys.exit('No out-of-sample matches in that window.')
    rng = np.random.default_rng(0)

    print(f"\n  Pool: {len(pool)} matches from {args.since}, "
          f"{pool['league'].nunique()} leagues, "
          f"probabilities = {'market where known' if not args.no_odds else 'model only'}")
    print(f"  Coupons: {args.n} x {args.size} matches, prize at "
          f"{args.threshold}+\n")
    print(f"  {'sampling':<28} {'N':>6} {'claimed':>9} {'actual':>9} "
          f"{'P(prize)':>11} {'realised':>9} {'sd':>7} {'95% CI':>8}")
    print('  ' + '-' * 92)

    qs, hits = sample_coupons(pool, args.size, args.n, False, rng)
    report('spread over the window', qs, hits, args.threshold, len(pool))
    qs2, hits2 = sample_coupons(pool, args.size, args.n, True, rng)
    report('same matchday', qs2, hits2, args.threshold, len(pool))

    print("\n  'claimed' is the sum of the picked probabilities (the "
          "Poisson-binomial mean); 'actual' is how many really landed.")
    print("  'P(prize)' is what the app would have shown; 'realised' is how "
          "often the coupon truly reached the threshold.")
    print("  The CI is on 'actual', derived from the underlying matches — the "
          "coupons are resampled from one pool, so they are not independent "
          "evidence.")
    if len(hits2):
        print(f"\n  Independence check: spread sd={hits.std(ddof=1):.2f} vs "
              f"same-day sd={hits2.std(ddof=1):.2f}. A materially larger "
              "same-day spread would mean matchdays are correlated, which "
              "would make P(prize) conservative at high thresholds.")
    print("\n  Reading this honestly: P(prize) is a tail probability over "
          "13-15 matches, so a per-match calibration error of a point or two — "
          "well inside the noise on any window this size — moves it by a large "
          "relative amount. Treat it as an estimate with real uncertainty, and "
          "let scripts/track.py accumulate your own coupons for the answer "
          "that actually counts.")


if __name__ == '__main__':
    main()
