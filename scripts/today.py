"""
Today's Predictions — one-command daily driver
===============================================
Fetches the upcoming-fixtures odds feed, predicts every club fixture it has
models for (market-anchored 1X2 when odds are available), and appends today's
World Cup matches from the international module.

Usage:
    python scripts/today.py              # next 3 days
    python scripts/today.py --days 7
    python scripts/today.py --csv out.csv
    python scripts/today.py --no-wc      # club fixtures only
"""

import argparse
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from scripts import config
from scripts import data_loader
from scripts import fixtures as fixtures_mod
from scripts import predict as predict_mod
from scripts import utils

warnings.filterwarnings('ignore')


def _pick(p: dict) -> tuple[str, float]:
    """Best 1X2 pick label + probability."""
    options = [('1', p['home']), ('X', p['draw']), ('2', p['away'])]
    return max(options, key=lambda x: x[1])


def club_section(days: int) -> list[dict]:
    fixtures_mod.fetch()
    fx = fixtures_mod.load(fetch_if_missing=False)

    now = pd.Timestamp.now().normalize()
    if not fx.empty:
        fx = fx[(fx['Date'] >= now)
                & (fx['Date'] <= now + pd.Timedelta(days=days))]
    if fx.empty:
        print(f"\n  No club fixtures in the feed for the next {days} days "
              "(between matchdays / league break).")
        return []
    fx = fx.sort_values(['Date', 'league'])

    print("Loading club data + models (this takes a moment)...")
    hist = data_loader.load_processed_data()
    team_stats = utils.get_team_stats_table(hist)
    team_to_league = utils.get_team_to_league_map(hist)

    rows = []
    for _, m in fx.iterrows():
        home, away, league = m['HomeTeam'], m['AwayTeam'], m['league']
        try:
            pred = predict_mod.predict_match(
                home, away, team_stats, team_to_league, hist,
                include_xg=False, prediction_date=m['Date'])
        except Exception:
            continue
        if '1x2' not in pred:
            continue

        p = pred['1x2']
        pick, prob = _pick(p)
        row = {
            'Date': m['Date'].date(),
            'League': config.LEAGUE_REGISTRY.get(league, {}).get(
                'display_name', league),
            'Home': home, 'Away': away,
            'P(1)': p['home'], 'P(X)': p['draw'], 'P(2)': p['away'],
            'Pick': pick, 'Conf': prob,
            'Anchored': 'market' in pred,
        }
        if 'ou25' in pred:
            row['P(O2.5)'] = pred['ou25']['over']
        if 'btts' in pred:
            row['P(BTTS)'] = pred['btts']['yes']
        # model-vs-market edge on the model's favourite outcome
        if 'market' in pred and '1x2_model' in pred:
            imp = pred['market']['implied']
            mod = pred['1x2_model']
            key = {'1': 'home', 'X': 'draw', '2': 'away'}[
                _pick(mod)[0]]
            row['Edge'] = mod[key] - imp[key]
        rows.append(row)
    return rows


def print_club_table(rows: list[dict]) -> None:
    if not rows:
        return
    print(f"\n  CLUB FIXTURES ({len(rows)} matches with models)\n")
    print(f"  {'Date':<11} {'League':<26} {'Match':<42} "
          f"{'1':>5} {'X':>5} {'2':>5}  {'Pick':<4} {'O2.5':>5} "
          f"{'BTTS':>5} {'Edge':>6}")
    print('  ' + '-' * 122)
    for r in rows:
        match = f"{r['Home']} vs {r['Away']}"
        anchor = '*' if r['Anchored'] else ' '
        edge = f"{r['Edge']:+.0%}" if 'Edge' in r else '     -'
        print(f"  {str(r['Date']):<11} {r['League']:<26} {match:<42} "
              f"{r['P(1)']:>5.0%} {r['P(X)']:>5.0%} {r['P(2)']:>5.0%} "
              f"{anchor}{r['Pick']:<4} {r.get('P(O2.5)', float('nan')):>5.0%} "
              f"{r.get('P(BTTS)', float('nan')):>5.0%} {edge:>6}")
    print("\n  * = 1X2 anchored to live bookmaker odds. Edge = raw model vs "
          "market on the model's pick\n      (only trust edges in leagues "
          "where the model beats the book — see blend.py evaluate).")


def wc_section(days: int) -> None:
    from scripts import international as intl

    # refresh results once every 12h so ratings include the latest matchday
    try:
        age = time.time() - os.path.getmtime(intl.INTL_FILE)
        if age > 12 * 3600:
            intl.cmd_update(None)
    except OSError:
        pass

    if not os.path.exists(intl.WC_FIXTURES_FILE):
        return
    fixtures = pd.read_csv(intl.WC_FIXTURES_FILE, parse_dates=['date'])
    now = pd.Timestamp.now().normalize()
    fixtures = fixtures[(fixtures['date'] >= now)
                        & (fixtures['date'] <= now + pd.Timedelta(days=days))]
    if fixtures.empty:
        return

    df = intl.load_results()
    ratings, hist = intl.run_elo(df)
    model = intl.GoalModel().fit(hist)

    print(f"\n  WORLD CUP ({len(fixtures)} matches, ratings through "
          f"{df['date'].max().date()})\n")
    print(f"  {'Date':<11} {'Match':<42} {'1':>5} {'X':>5} {'2':>5} "
          f"{'O2.5':>6} {'BTTS':>6}")
    print('  ' + '-' * 86)
    for _, m in fixtures.sort_values('date').iterrows():
        h, a = m['home_team'], m['away_team']
        if h not in ratings or a not in ratings:
            continue
        p = model.market_probs(ratings[h], ratings[a],
                               neutral=bool(m['neutral']))
        match = f"{h} vs {a}"
        print(f"  {str(m['date'].date()):<11} {match:<42} "
              f"{p['p_home']:>5.0%} {p['p_draw']:>5.0%} {p['p_away']:>5.0%} "
              f"{p['p_over25']:>6.0%} {p['p_btts']:>6.0%}")


def main():
    parser = argparse.ArgumentParser(description="Today's predictions")
    parser.add_argument('--days', type=int, default=3,
                        help='Days ahead to include (default 3)')
    parser.add_argument('--csv', type=str, default=None,
                        help='Also write the club table to a CSV file')
    parser.add_argument('--no-wc', action='store_true',
                        help='Skip the World Cup section')
    args = parser.parse_args()

    rows = club_section(args.days)
    print_club_table(rows)
    if rows and args.csv:
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"\n  Saved -> {args.csv}")

    if not args.no_wc:
        try:
            wc_section(args.days)
        except Exception as e:
            print(f"\n  (World Cup section unavailable: {e})")


if __name__ == '__main__':
    main()
