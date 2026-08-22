"""
Upcoming Fixtures + Market Odds
================================
football-data.co.uk publishes odds for forthcoming matches:
  fixtures.csv             rich European leagues (Div codes: E0, D1, SP1, ...)
  new_league_fixtures.csv  extra leagues (Country/League names)

This module downloads both into data/_fixtures/ and exposes a lookup so
predict.py can anchor its 1X2 probabilities to the market (see blend.py for
how the weights were fitted and validated).

CLI:
    python scripts/fixtures.py          # download + show what's available
"""

import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from scripts import config
from scripts import data_loader

FIXTURES_DIR = os.path.join(config.DATA_DIR, '_fixtures')
SOURCES = {
    'rich': 'https://www.football-data.co.uk/fixtures.csv',
    'extra': 'https://www.football-data.co.uk/new_league_fixtures.csv',
}
MAX_AGE_HOURS = 12

# Div code -> league key, derived from the FETCH_SOURCES season-file URLs.
DIV_TO_LEAGUE = {
    m.group(1): league
    for league, src in config.FETCH_SOURCES.items()
    if (m := re.search(r'/mmz4281/\{season\}/(\w+)\.csv', src['url']))
}

# Country name (new_league_fixtures.csv) -> league key. Derived from
# config.SPARSE_COUNTRY rather than maintained by hand — a hand-kept copy
# silently missed every sparse league added after it was written, so their
# fixtures never appeared in the feed.
COUNTRY_TO_LEAGUE = {country: league
                     for league, country in config.SPARSE_COUNTRY.items()}


def fetch(force: bool = False) -> None:
    """Download both fixture files unless a fresh copy already exists."""
    import requests
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    for name, url in SOURCES.items():
        path = os.path.join(FIXTURES_DIR, f'{name}.csv')
        if (not force and os.path.exists(path)
                and time.time() - os.path.getmtime(path) < MAX_AGE_HOURS * 3600):
            continue
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with open(path, 'wb') as f:
            f.write(resp.content)


def load(fetch_if_missing: bool = True) -> pd.DataFrame:
    """Return upcoming fixtures with uniform columns:
    league, Date, HomeTeam, AwayTeam, OddsH/D/A, OddsOver25/Under25.
    """
    frames = []

    rich_path = os.path.join(FIXTURES_DIR, 'rich.csv')
    extra_path = os.path.join(FIXTURES_DIR, 'extra.csv')
    if fetch_if_missing and not (os.path.exists(rich_path)
                                 and os.path.exists(extra_path)):
        fetch()

    if os.path.exists(rich_path):
        df = pd.read_csv(rich_path, encoding='utf-8-sig', on_bad_lines='skip')
        df['league'] = df['Div'].map(DIV_TO_LEAGUE)
        frames.append(df)

    if os.path.exists(extra_path):
        df = pd.read_csv(extra_path, encoding='utf-8-sig', on_bad_lines='skip')
        df['league'] = df['Country'].map(COUNTRY_TO_LEAGUE)
        df = data_loader.normalize_columns(df)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=['league', 'HomeTeam', 'AwayTeam'])
    out = data_loader.extract_odds(out)
    out['Date'] = pd.to_datetime(out['Date'], format='%d/%m/%Y', errors='coerce')
    cols = ['league', 'Date', 'HomeTeam', 'AwayTeam',
            'OddsH', 'OddsD', 'OddsA', 'OddsOver25', 'OddsUnder25']
    return out[cols]


_CACHE: pd.DataFrame | None = None


def load_cached() -> pd.DataFrame:
    global _CACHE
    if _CACHE is None:
        _CACHE = load()
    return _CACHE


def lookup_odds(fixtures: pd.DataFrame, league: str,
                home: str, away: str) -> dict | None:
    """Find market odds for one upcoming fixture; None if not listed."""
    if fixtures is None or fixtures.empty:
        return None
    fresh = fixtures['Date'] >= pd.Timestamp.now() - pd.Timedelta(hours=36)
    m = fixtures[fresh
                 & (fixtures['league'] == league)
                 & (fixtures['HomeTeam'] == home)
                 & (fixtures['AwayTeam'] == away)]
    if m.empty:
        return None
    r = m.iloc[-1]
    if pd.isna(r['OddsH']) or pd.isna(r['OddsD']) or pd.isna(r['OddsA']):
        return None
    out = {'OddsH': float(r['OddsH']), 'OddsD': float(r['OddsD']),
           'OddsA': float(r['OddsA']), 'Date': r['Date']}
    if pd.notna(r['OddsOver25']) and pd.notna(r['OddsUnder25']):
        out['OddsOver25'] = float(r['OddsOver25'])
        out['OddsUnder25'] = float(r['OddsUnder25'])
    return out


def find_odds_any_league(fixtures: pd.DataFrame, home: str, away: str,
                         hours_back: int = 36) -> dict | None:
    """Odds for an upcoming fixture, matched by team name across all leagues.

    ``lookup_odds`` needs the league key and exact names; a coupon row has
    neither, so this matches accent-folded names (and the coupon aliases) over
    the whole feed. Used to fill in a coupon's odds automatically.
    """
    if fixtures is None or fixtures.empty:
        return None
    from scripts import toto

    h, a = toto._fold(toto.apply_alias(home)), toto._fold(toto.apply_alias(away))
    if not h or not a:
        return None
    fresh = fixtures[fixtures['Date']
                     >= pd.Timestamp.now() - pd.Timedelta(hours=hours_back)]
    if fresh.empty:
        return None
    fh = fresh['HomeTeam'].map(lambda x: toto._fold(x))
    fa = fresh['AwayTeam'].map(lambda x: toto._fold(x))

    m = fresh[(fh == h) & (fa == a)]
    if m.empty:                                   # fall back to containment
        m = fresh[[bool(x) and bool(y) for x, y in zip(
            fh.map(lambda v: h in v or v in h), fa.map(lambda v: a in v or v in a))]]
    if m.empty:
        return None
    r = m.iloc[0]
    if pd.isna(r['OddsH']) or pd.isna(r['OddsD']) or pd.isna(r['OddsA']):
        return None
    return {'OddsH': float(r['OddsH']), 'OddsD': float(r['OddsD']),
            'OddsA': float(r['OddsA']), 'Date': r['Date'],
            'HomeTeam': r['HomeTeam'], 'AwayTeam': r['AwayTeam'],
            'league': r.get('league')}


def main():
    fetch(force=True)
    fx = load(fetch_if_missing=False)
    if fx.empty:
        print("No fixtures available.")
        return
    with_odds = fx.dropna(subset=['OddsH'])
    print(f"\n  {len(fx)} upcoming fixtures, {len(with_odds)} with 1X2 odds, "
          f"{fx['Date'].min().date()} to {fx['Date'].max().date()}\n")
    for league, g in with_odds.groupby('league'):
        disp = config.LEAGUE_REGISTRY.get(league, {}).get('display_name', league)
        print(f"  {disp:<28} {len(g):>4} fixtures")


if __name__ == '__main__':
    main()
