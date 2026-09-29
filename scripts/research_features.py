"""Day-batched research features, shared by replay and future predictions.

This deliberately does not load production features, tuned Elo parameters or
models. Results on a date become available only after every fixture that day
has been featurized. It is a conservative date-level replay, not an intraday
availability reconstruction. Missing statistics remain missing, not zero.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import re

import numpy as np
import pandas as pd

from scripts import config, data_loader

VERSION = 'causal-day-state-v1'
MARKET_VERSION = VERSION + '+mkt2'     # FeatureState(market=True)
# Prices a finished match is rated from, best first. Closing prices are
# allowed HERE because the update runs after the match is complete: they are
# history by then, like the score. (99.9% of matches carry the average close;
# pre-match prices exist for only the rich leagues.)
RATING_ODDS = [('AvgCH', 'AvgCD', 'AvgCA'), ('PSCH', 'PSCD', 'PSCA'),
               ('B365CH', 'B365CD', 'B365CA')]
MKT_K = 40.0              # prices are far less noisy than results: move faster
MKT_RETENTION = 0.9
MKT_HOME_ADV = 60.0
CALENDAR_LEAGUES = {'argentina', 'BRAZIL', 'chn', 'fin', 'japan', 'norsk',
                   'irish', 'swedish', 'usa'}
KEY = ['league', 'Date', 'HomeTeam', 'AwayTeam']
PRE_ODDS = [('AvgH', 'AvgD', 'AvgA'), ('B365H', 'B365D', 'B365A'),
            ('PSH', 'PSD', 'PSA'), ('PH', 'PD', 'PA')]


def season_id(league, date, supplied=None):
    if supplied is not None and pd.notna(supplied):
        return str(supplied).removesuffix('.0')
    date = pd.Timestamp(date)
    start = date.year if league in CALENDAR_LEAGUES or date.month >= 7 else date.year - 1
    return str(start) if league in CALENDAR_LEAGUES else f'{start}/{start+1}'


def attach_odds(df):
    """Preserve provenance and never substitute closing prices for pre-close.

    These historical snapshots have no per-quote timestamps. They support
    probability benchmarking; returns at them are hypothetical, not executable.
    Research matrices use class order [away, draw, home].
    """
    out = df.copy()
    for col in ['qA', 'qD', 'qH', 'betA', 'betD', 'betH']:
        out[col] = np.nan
    out['odds_source'] = ''
    for triple in PRE_ODDS:
        if not set(triple).issubset(out.columns):
            continue
        odds = out[list(triple)].apply(pd.to_numeric, errors='coerce')
        ok = (np.isfinite(odds).all(axis=1) & odds.gt(1).all(axis=1)
              & out['qH'].isna())
        inv = 1 / odds.loc[ok].to_numpy(float)
        out.loc[ok, ['qA', 'qD', 'qH']] = (inv / inv.sum(axis=1, keepdims=True))[:, ::-1]
        out.loc[ok, 'odds_source'] = '/'.join(triple)
    target = ['B365A', 'B365D', 'B365H']
    if set(target).issubset(out.columns):
        odds = out[target].apply(pd.to_numeric, errors='coerce')
        ok = np.isfinite(odds).all(axis=1) & odds.gt(1).all(axis=1)
        out.loc[ok, ['betA', 'betD', 'betH']] = odds.loc[ok].to_numpy()
    return out


def load_raw(leagues=None):
    """Read raw source files with explicit seasons, hashes and conflict checks."""
    leagues = list(leagues or config.LEAGUE_REGISTRY)
    unknown = set(leagues) - set(config.LEAGUE_REGISTRY)
    if unknown:
        raise ValueError(f'Unknown leagues: {sorted(unknown)}')
    frames, manifest = [], []
    for league in sorted(leagues):
        for path in sorted(Path(config.DATA_DIR, league).glob('*.csv')):
            payload = path.read_bytes()
            manifest.append({'path': str(path.relative_to(config.PROJECT_ROOT)),
                             'sha256': hashlib.sha256(payload).hexdigest()})
            try:
                df = pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(path, encoding='latin1', low_memory=False)
            df = data_loader.normalize_columns(df)
            if not set(config.REQUIRED_COLS).issubset(df.columns):
                raise ValueError(f'Missing required columns: {path}')
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, format='mixed', errors='coerce').dt.normalize()
            for col in ['FTHG', 'FTAG'] + config.RICH_COLS:
                if col in df:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            # A partially populated fixture is not a completed training row.
            df = df.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'])
            df = df[df.Date.dt.year >= 2015].copy()
            goals = df[['FTHG', 'FTAG']].to_numpy(float)
            if (not np.isfinite(goals).all() or (goals < 0).any()
                    or (goals != np.floor(goals)).any()):
                raise ValueError(f'Invalid final scores: {path}')
            expected = np.where(df.FTHG > df.FTAG, 'H', np.where(df.FTHG < df.FTAG, 'A', 'D'))
            if not np.all(df.FTR.to_numpy() == expected):
                raise ValueError(f'Result does not match final score: {path}')
            df['league'] = league
            stamp = re.fullmatch(r'[A-Z0-9]+_(\d{2})(\d{2})\.csv', path.name)
            if stamp and config.LEAGUE_REGISTRY[league]['type'] == 'rich':
                df['SeasonId'] = f'20{stamp[1]}/20{stamp[2]}'
            else:
                supplied = df.get('Season', pd.Series(None, index=df.index, dtype=object))
                df['SeasonId'] = [season_id(league, d, s) for d, s in zip(df.Date, supplied)]
            frames.append(attach_odds(df))
    if not frames:
        raise ValueError('No raw matches found')
    out = pd.concat(frames, ignore_index=True)
    dup = out[out.duplicated(KEY, keep=False)]
    if len(dup) and (dup.groupby(KEY)[['FTHG', 'FTAG', 'SeasonId']].nunique() > 1).any().any():
        raise ValueError('Conflicting duplicate results or season IDs; reconcile sources first')
    out = out.drop_duplicates(KEY).sort_values(KEY, kind='stable').reset_index(drop=True)
    out['match_id'] = [hashlib.sha256('|'.join(map(str, row)).encode()).hexdigest()[:20]
                       for row in out[KEY].itertuples(index=False, name=None)]
    return out, manifest


@dataclass
class Team:
    rating: float = 1500.0
    season: str | None = None
    last_date: pd.Timestamp | None = None
    games: int = 0
    season_games: int = 0
    season_points: int = 0
    matches: deque = field(default_factory=lambda: deque(maxlen=10))
    home: deque = field(default_factory=lambda: deque(maxlen=10))
    away: deque = field(default_factory=lambda: deque(maxlen=10))
    # market block (FeatureState(market=True) only): a rating fitted to past
    # pre-match prices instead of results, and what those prices said
    mkt_rating: float = 1500.0
    mkt_season: str | None = None
    mkt_n: int = 0
    mkt_hist: deque = field(default_factory=lambda: deque(maxlen=10))


def mean(values):
    values = np.asarray(list(values), dtype=float)
    valid = values[np.isfinite(values)]
    return float(valid.mean()) if len(valid) else np.nan


class FeatureState:
    """One source of truth for historical and live fixture features.

    League-scoped identities make promotion a visible cold start. No ratings
    are silently transferred between incompatible league strength scales.
    Feature lookup is read-only, including season regression.
    """
    def __init__(self, k=20.0, season_retention=0.85, market=False,
                 mkt_k=MKT_K, mkt_retention=MKT_RETENTION):
        if not 0 <= season_retention <= 1 or k <= 0:
            raise ValueError('Invalid Elo settings')
        self.k, self.season_retention = float(k), float(season_retention)
        self.market = bool(market)
        self.mkt_k, self.mkt_retention = float(mkt_k), float(mkt_retention)
        self.teams = {}
        self.league_results = defaultdict(lambda: deque(maxlen=150))
        self.h2h = defaultdict(lambda: deque(maxlen=6))
        self.last_day = None

    def _team(self, league, name):
        return self.teams.get((league, name), Team())

    def _rating(self, team, season):
        return (1500 + self.season_retention * (team.rating - 1500)
                if team.season is not None and team.season != season else team.rating)

    def fixture(self, row):
        day, league = pd.Timestamp(row['Date']).normalize(), row['league']
        if self.last_day is not None and day <= self.last_day:
            raise ValueError('Fixture must be after every completed day in state')
        season = season_id(league, day, row.get('SeasonId'))
        h, a = self._team(league, row['HomeTeam']), self._team(league, row['AwayTeam'])
        result = {'EloDiff': self._rating(h, season) - self._rating(a, season)}
        for prefix, team, venue in [('Home', h, h.home), ('Away', a, a.away)]:
            hist = list(team.matches)
            result[prefix+'HistoryN'] = min(team.games, 50)
            result[prefix+'SeasonN'] = team.season_games if team.season == season else 0
            result[prefix+'SeasonPPG'] = (team.season_points / team.season_games
                if team.season == season and team.season_games else np.nan)
            result[prefix+'Rest'] = min((day-team.last_date).days, 90) if team.last_date is not None else np.nan
            for window in (3, 10):
                for key in ['gf', 'ga', 'points', 'sot', 'corners', 'cards', 'opponent']:
                    result[f'{prefix}_{key}_{window}'] = mean(m[key] for m in hist[-window:])
            for key in ['sot', 'corners', 'cards']:
                result[f'{prefix}_{key}_observed'] = sum(np.isfinite(m[key]) for m in hist)
            for key in ['gf', 'ga', 'points']:
                result[f'{prefix}_venue_{key}'] = mean(m[key] for m in venue)
            result[prefix+'CleanSheets'] = mean(m['ga'] == 0 for m in hist)
            result[prefix+'BTTS'] = mean(m['gf'] > 0 and m['ga'] > 0 for m in hist)
        history = self.h2h.get((league, *sorted([row['HomeTeam'], row['AwayTeam']])), [])
        result['H2HN'] = len(history)
        result['H2HGoalDiff'] = mean((hg-ag if home == row['HomeTeam'] else ag-hg)
                                   for home, hg, ag in history)
        result['LeagueHomeRate'] = mean(self.league_results.get(league, []))
        if self.market:
            result.update(self._market_features(h, a, season))
        return result

    # ------------------------------------------------------------------ market
    # Past prices are facts once a match is played, so a rating fitted to them
    # is as causal as Elo. It carries the market's opinion of a team into
    # fixtures that have no price of their own, which is exactly where the
    # model is used. A fixture's own prices are never read: the rating only
    # moves in update_day, after the match is finished.
    def _mkt_rating(self, team, season):
        return (1500 + self.mkt_retention * (team.mkt_rating - 1500)
                if team.mkt_season is not None and team.mkt_season != season
                else team.mkt_rating)

    def _market_features(self, h, a, season):
        out = {'MktRatingDiff': (self._mkt_rating(h, season) - self._mkt_rating(a, season)
                                 if h.mkt_n and a.mkt_n else np.nan)}
        for prefix, team in (('Home', h), ('Away', a)):
            out[prefix + 'MktN'] = min(team.mkt_n, 50)
            out[prefix + '_mktdraw_10'] = mean(d for d, _ in team.mkt_hist)
            out[prefix + '_mktresid_10'] = mean(r for _, r in team.mkt_hist)
        return out

    @staticmethod
    def _rating_probs(row):
        """Margin-free (A, D, H) from the best available price, or None."""
        for triple in RATING_ODDS:
            try:
                odds = np.array([float(row.get(c, np.nan)) for c in triple])
            except (TypeError, ValueError):
                continue
            if np.isfinite(odds).all() and (odds > 1).all():
                inv = 1 / odds
                ph, pd_, pa = inv / inv.sum()
                return pa, pd_, ph
        q = [row.get(c, np.nan) for c in ('qA', 'qD', 'qH')]
        return tuple(q) if np.all(np.isfinite(q)) else None

    def _market_update(self, row, h, a, season, hg, ag):
        q = self._rating_probs(row)
        if q is None:
            return
        qa, qd, qh = q
        hr, ar = self._mkt_rating(h, season), self._mkt_rating(a, season)
        expected = 1 / (1 + 10 ** ((ar - hr - MKT_HOME_ADV) / 400))
        delta = self.mkt_k * ((qh + .5 * qd) - expected)
        h_pts = 3 if hg > ag else int(hg == ag)
        a_pts = 3 if ag > hg else int(hg == ag)
        for team, rating, pts, exp_pts in (
                (h, hr + delta, h_pts, 3 * qh + qd),
                (a, ar - delta, a_pts, 3 * qa + qd)):
            team.mkt_rating, team.mkt_season = rating, season
            team.mkt_n += 1
            team.mkt_hist.append((qd, pts - exp_pts))

    def update_day(self, block):
        days = pd.to_datetime(block['Date']).dt.normalize().unique()
        if len(days) != 1 or (self.last_day is not None and pd.Timestamp(days[0]) <= self.last_day):
            raise ValueError('Update exactly one strictly later completed day')
        # With day-level records, a team cannot safely have two completed games
        # on a day: reject rather than invent a within-day observation order.
        seen = set()
        rows = block.sort_values(KEY, kind='stable').to_dict('records')
        for row in rows:
            for name in [row['HomeTeam'], row['AwayTeam']]:
                key = (row['league'], name)
                if key in seen:
                    raise ValueError(f'Team plays twice on one date: {key}')
                seen.add(key)
        for row in rows:
            league, day = row['league'], pd.Timestamp(row['Date']).normalize()
            season = season_id(league, day, row.get('SeasonId'))
            h = self.teams.setdefault((league, row['HomeTeam']), Team())
            a = self.teams.setdefault((league, row['AwayTeam']), Team())
            hr, ar = self._rating(h, season), self._rating(a, season)
            hg, ag = float(row['FTHG']), float(row['FTAG'])
            if not np.isfinite([hg, ag]).all() or min(hg, ag) < 0:
                raise ValueError('Cannot update from an unplayed or invalid result')
            if self.market:
                self._market_update(row, h, a, season, hg, ag)
            win = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
            delta = self.k * (win - 1/(1+10**((ar-hr-60)/400)))
            for team, prefix, gf, ga, rating, opponent, points, venue in [
                (h, 'H', hg, ag, hr+delta, ar, 3 if win == 1 else int(win == .5), h.home),
                (a, 'A', ag, hg, ar-delta, hr, 3 if win == 0 else int(win == .5), a.away),
            ]:
                if team.season != season:
                    team.season_games = team.season_points = 0
                team.season = season
                team.season_games += 1
                team.season_points += points
                team.games += 1
                team.last_date, team.rating = day, rating
                yc, rc = row.get(prefix+'Y', np.nan), row.get(prefix+'R', np.nan)
                record = dict(gf=gf, ga=ga, points=points, opponent=opponent,
                              sot=row.get(prefix+'ST', np.nan), corners=row.get(prefix+'C', np.nan),
                              cards=yc+2*rc if pd.notna(yc) and pd.notna(rc) else np.nan)
                team.matches.append(record)
                venue.append(record)
            self.league_results[league].append(int(hg > ag))
            self.h2h[(league, *sorted([row['HomeTeam'], row['AwayTeam']]))].append((row['HomeTeam'], hg, ag))
        self.last_day = pd.Timestamp(days[0])


def replay(raw, state=None):
    state = state or FeatureState()
    frames = []
    for _, day in raw.sort_values(['Date']+KEY[:1]+KEY[2:], kind='stable').groupby('Date', sort=True):
        feats = pd.DataFrame([state.fixture(row) for row in day.to_dict('records')], index=day.index)
        frames.append(pd.concat([day, feats], axis=1))
        state.update_day(day)
    if not frames:
        raise ValueError('No matches to replay')
    return pd.concat(frames).reset_index(drop=True), state


def feature_columns(market=False):
    dummy = dict(Date='2000-01-01', league='british_pl', HomeTeam='H', AwayTeam='A')
    return list(FeatureState(market=market).fixture(dummy))


def version(market=False):
    return MARKET_VERSION if market else VERSION
