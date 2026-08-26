"""
European club predictor (Champions League / Europa League / Conference League)
==============================================================================
Cross-league club matches cannot be priced by the per-league models: their Elo
is computed per league in isolation (every league's ruler starts at 1500), and
the training data contains no inter-league matches at all.

This module puts every European club on ONE ruler by using clubelo.com's free
API (globally calibrated club Elo, updated after every match, 55 countries),
and learns the mapping Elo-difference -> goals/1X2 from our own domestic
matches — legitimate because within a league both clubs sit on the same global
ruler too, so the curve carries over to cross-league fixtures. Same two-head
design the international module validated on 11k matches: a Poisson goal model
for the score grid plus a direct softmax 1X2 head, log-pooled.

    python scripts/clubs_europe.py update            # refresh ratings + history
    python scripts/clubs_europe.py ratings [--top 30]
    python scripts/clubs_europe.py predict --home "Galatasaray" --away "Real Madrid"
    python scripts/clubs_europe.py backtest          # OOS sanity vs book

Used automatically by the Toto tab: a row whose clubs are in two different
leagues (or whose league models don't know them) falls through to this model
before the national-team fallback.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from scripts import config

CLUBELO_DIR = os.path.join(config.DATA_DIR, 'global', 'clubelo')
API = 'http://api.clubelo.com/{date}'
LATEST_MAX_AGE_DAYS = 3          # refetch the current snapshot after this
HISTORY_MONTHS = 24              # monthly snapshots used to fit the goal model
MIX_MULTINOM = 0.60              # weight of the softmax head (validated intl)
MAX_GOALS = 10

_CACHE: dict = {}


# =============================================================================
# RATINGS (clubelo.com snapshots)
# =============================================================================
def _snapshot_path(date_str: str) -> str:
    return os.path.join(CLUBELO_DIR, f'{date_str}.csv')


def _fetch_snapshot(date_str: str, timeout: int = 30) -> pd.DataFrame | None:
    """Download one full-day ratings snapshot and cache it to disk."""
    import requests
    try:
        r = requests.get(API.format(date=date_str), timeout=timeout)
        r.raise_for_status()
    except Exception:
        return None
    text = r.text
    if 'Club' not in text.split('\n', 1)[0]:
        return None
    os.makedirs(CLUBELO_DIR, exist_ok=True)
    with open(_snapshot_path(date_str), 'w', encoding='utf-8') as f:
        f.write(text)
    return _read_snapshot(date_str)


def _read_snapshot(date_str: str) -> pd.DataFrame | None:
    path = _snapshot_path(date_str)
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    if 'Club' not in df.columns or 'Elo' not in df.columns:
        return None
    return df[['Club', 'Country', 'Level', 'Elo']].dropna(subset=['Club', 'Elo'])


def current_ratings(fetch_if_stale: bool = True) -> pd.DataFrame | None:
    """Latest snapshot (cached; refetched when older than LATEST_MAX_AGE_DAYS)."""
    if 'current' in _CACHE:
        return _CACHE['current']
    today = pd.Timestamp.today().normalize()
    # newest cached snapshot within the freshness window, else fetch today's
    if os.path.isdir(CLUBELO_DIR):
        cached = sorted(f[:-4] for f in os.listdir(CLUBELO_DIR)
                        if f.endswith('.csv'))
        for d in reversed(cached):
            try:
                age = (today - pd.Timestamp(d)).days
            except ValueError:
                continue
            if age <= LATEST_MAX_AGE_DAYS:
                _CACHE['current'] = _read_snapshot(d)
                return _CACHE['current']
    df = None
    if fetch_if_stale:
        df = _fetch_snapshot(today.strftime('%Y-%m-%d'))
    if df is None and os.path.isdir(CLUBELO_DIR):     # offline: newest cached
        cached = sorted(f[:-4] for f in os.listdir(CLUBELO_DIR)
                        if f.endswith('.csv'))
        if cached:
            df = _read_snapshot(cached[-1])
    _CACHE['current'] = df
    return df


def month_starts(n: int = HISTORY_MONTHS) -> list[str]:
    end = pd.Timestamp.today().normalize().replace(day=1)
    return [(end - pd.DateOffset(months=k)).strftime('%Y-%m-%d')
            for k in range(n, 0, -1)]


def ensure_history(verbose: bool = False) -> list[str]:
    """Make sure the monthly fitting snapshots exist locally; returns dates."""
    have = []
    for d in month_starts():
        if os.path.isfile(_snapshot_path(d)) or _fetch_snapshot(d) is not None:
            have.append(d)
            if verbose:
                print(f'  snapshot {d} ok')
        elif verbose:
            print(f'  snapshot {d} FAILED')
    return have


# =============================================================================
# NAME RESOLUTION (football-data / user names -> clubelo names)
# =============================================================================
# clubelo transliterates umlauts as oe/ue (Goeztepe, Malmoe, Bodoe Glimt);
# folding both sides through this makes those match plain typing.
def _fold_translit(name: str) -> str:
    from scripts.toto import _fold
    s = _fold(name)
    return s.replace('oe', 'o').replace('ue', 'u')


# Names neither folding nor substrings can bridge.
CLUBELO_ALIASES = {
    'fcsb': 'Steaua', 'copenhagen': 'FC Kobenhavn',
    'fc copenhagen': 'FC Kobenhavn', 'qarabag': 'Karabakh Agdam',
    'red star': 'Crvena Zvezda', 'red star belgrade': 'Crvena Zvezda',
    'dinamo kiev': 'Dynamo Kyiv', 'dynamo kiev': 'Dynamo Kyiv',
    'basaksehir': 'Bueyueksehir', 'istanbul basaksehir': 'Bueyueksehir',
    'bodo/glimt': 'Bodoe Glimt', 'bodo glimt': 'Bodoe Glimt',
    'sporting': 'Sporting', 'sporting lisbon': 'Sporting',
    'atletico madrid': 'Atletico', 'ath madrid': 'Atletico',
    'psv eindhoven': 'PSV', 'man united': 'Man United',
    'inter milan': 'Inter', 'ac milan': 'Milan',
    'club brugge': 'Brugge', 'dinamo zagreb': 'Dinamo Zagreb',
}


def _index(ratings: pd.DataFrame) -> dict:
    key = 'name_index'
    if key not in _CACHE:
        _CACHE[key] = {_fold_translit(c): c for c in ratings['Club']}
    return _CACHE[key]


def resolve(name: str, ratings: pd.DataFrame | None = None) -> str | None:
    """clubelo club name for a user/football-data name, or None."""
    if ratings is None:
        ratings = current_ratings()
    if ratings is None or not str(name).strip():
        return None
    from scripts.toto import apply_alias, _fold
    raw = str(name).strip()
    alias = CLUBELO_ALIASES.get(_fold(raw))
    if alias:
        raw = alias
    else:
        raw = apply_alias(raw)                 # reuse the coupon alias table
    idx = _index(ratings)
    k = _fold_translit(raw)
    if k in idx:
        return idx[k]
    hits = [orig for f, orig in idx.items() if k and (k in f or f in k)]
    return hits[0] if len(hits) == 1 else None


def elo_of(name: str, ratings: pd.DataFrame | None = None) -> float | None:
    if ratings is None:
        ratings = current_ratings()
    club = resolve(name, ratings)
    if club is None:
        return None
    row = ratings[ratings['Club'] == club]
    return float(row['Elo'].iloc[0]) if len(row) else None


# =============================================================================
# GOAL MODEL (Elo difference -> goals and 1X2), fitted on domestic matches
# =============================================================================
class EuroGoalModel:
    """Two heads on the global-Elo difference, log-pooled for 1X2."""

    def __init__(self):
        self.coef_ = None      # Poisson: [1, edge/400, is_home]
        self.W_ = None         # softmax: [1, e, tanh e, is_home, |e|] -> 3

    @staticmethod
    def _design_pois(edge, is_home):
        edge = np.atleast_1d(np.asarray(edge, float))
        return np.column_stack([np.ones(len(edge)), edge / 400.0,
                                np.asarray(is_home, float) * np.ones(len(edge))])

    @staticmethod
    def _design_1x2(edge, is_home):
        e = np.atleast_1d(np.asarray(edge, float)) / 400.0
        return np.column_stack([np.ones(len(e)), e, np.tanh(e),
                                np.asarray(is_home, float) * np.ones(len(e)),
                                np.abs(e)])

    def fit(self, edge, hg, ag, neutral=None):
        edge = np.asarray(edge, float)
        hg = np.asarray(hg, float)
        ag = np.asarray(ag, float)
        home = np.ones(len(edge)) if neutral is None \
            else (~np.asarray(neutral, bool)).astype(float)
        X = np.vstack([self._design_pois(edge, home),
                       self._design_pois(-edge, np.zeros(len(edge)))])
        y = np.concatenate([hg, ag])
        self.coef_ = self._irls(X, y)
        gd = hg - ag
        out = np.where(gd > 0, 0, np.where(gd == 0, 1, 2))
        self.W_ = self._softmax_fit(self._design_1x2(edge, home), out)
        return self

    @staticmethod
    def _irls(X, y, n_iter=60, tol=1e-9):
        beta = np.zeros(X.shape[1])
        beta[0] = math.log(max(y.mean(), 1e-3))
        for _ in range(n_iter):
            eta = X @ beta
            mu = np.exp(np.clip(eta, -20, 5))
            W = mu
            z = eta + (y - mu) / np.maximum(mu, 1e-9)
            XtW = X.T * W
            try:
                nb = np.linalg.solve(XtW @ X, XtW @ z)
            except np.linalg.LinAlgError:
                break
            if np.max(np.abs(nb - beta)) < tol:
                return nb
            beta = nb
        return beta

    @staticmethod
    def _softmax_fit(X, y, iters=300, l2=1e-4, step=8.0):
        n, p = X.shape
        W = np.zeros((p, 3))
        Y = np.zeros((n, 3)); Y[np.arange(n), y] = 1.0
        lr, prev = 1.0, np.inf
        for _ in range(iters):
            Z = X @ W
            Z -= Z.max(axis=1, keepdims=True)
            P = np.exp(Z); P /= P.sum(axis=1, keepdims=True)
            loss = -(Y * np.log(np.clip(P, 1e-12, None))).sum() / n \
                + l2 * (W[1:] ** 2).sum()
            G = X.T @ (P - Y) / n
            G[1:] += 2 * l2 * W[1:]
            if loss > prev:
                lr *= 0.5
            prev = loss
            W = W - lr * step * G
        return W

    # ------------------------------------------------------------- inference
    def lambdas(self, edge, home=True):
        xh = self._design_pois([edge], [home])
        xa = self._design_pois([-edge], [False])
        return (float(np.exp(np.clip(xh @ self.coef_, -20, 5))[0]),
                float(np.exp(np.clip(xa @ self.coef_, -20, 5))[0]))

    def market_probs(self, elo_home, elo_away, neutral=False):
        edge = float(elo_home) - float(elo_away)
        lam_h, lam_a = self.lambdas(edge, home=not neutral)
        g = np.arange(MAX_GOALS + 1)
        fact = np.array([math.factorial(int(i)) for i in g])
        grid = np.outer(np.exp(-lam_h) * lam_h ** g / fact,
                        np.exp(-lam_a) * lam_a ** g / fact)
        grid /= grid.sum()
        i, j = np.indices(grid.shape)
        regions = [i > j, i == j, i < j]
        p_grid = np.array([grid[r].sum() for r in regions])

        z = self._design_1x2([edge], [not neutral]) @ self.W_
        z -= z.max()
        p_dir = np.exp(z).ravel(); p_dir /= p_dir.sum()

        lg = (MIX_MULTINOM * np.log(np.clip(p_dir, 1e-12, None))
              + (1 - MIX_MULTINOM) * np.log(np.clip(p_grid, 1e-12, None)))
        lg -= lg.max()
        p = np.exp(lg); p /= p.sum()
        for r, target, base in zip(regions, p, p_grid):
            if base > 1e-12:
                grid[r] *= target / base
        grid /= grid.sum()
        total = i + j
        return {
            'lambda_home': lam_h, 'lambda_away': lam_a,
            'p_home': float(p[0]), 'p_draw': float(p[1]), 'p_away': float(p[2]),
            'p_over25': float(grid[total > 2.5].sum()),
            'p_btts': float(grid[(i > 0) & (j > 0)].sum()),
        }


# =============================================================================
# FITTING DATA (domestic matches joined to the monthly snapshots)
# =============================================================================
def build_training(verbose: bool = False) -> pd.DataFrame:
    """Domestic matches with both clubs' global Elo as of the prior snapshot."""
    dates = ensure_history(verbose=verbose)
    if not dates:
        return pd.DataFrame()
    snaps = {}
    for d in dates:
        s = _read_snapshot(d)
        if s is not None:
            snaps[pd.Timestamp(d)] = {_fold_translit(c): e for c, e in
                                      zip(s['Club'], s['Elo'])}
    if not snaps:
        return pd.DataFrame()
    snap_dates = sorted(snaps)

    df = pd.read_csv(config.PROCESSED_DATA_FILE,
                     usecols=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'],
                     parse_dates=['Date'])
    df = df[df['Date'] >= snap_dates[0]]

    # map football-data names once, via the current index (names are stable)
    ratings = current_ratings()
    if ratings is None:
        return pd.DataFrame()
    name_map = {}
    for t in set(df['HomeTeam']) | set(df['AwayTeam']):
        club = resolve(t, ratings)
        if club is not None:
            name_map[t] = _fold_translit(club)

    idx = np.searchsorted(pd.DatetimeIndex(snap_dates),
                          df['Date'].to_numpy()) - 1
    out_edge, out_hg, out_ag = [], [], []
    dts = df['Date'].to_numpy()
    hs = df['HomeTeam'].to_numpy(); as_ = df['AwayTeam'].to_numpy()
    hg = df['FTHG'].to_numpy(); ag = df['FTAG'].to_numpy()
    for k in range(len(df)):
        si = idx[k]
        if si < 0:
            continue
        snap = snaps[snap_dates[si]]
        hkey = name_map.get(hs[k]); akey = name_map.get(as_[k])
        if hkey is None or akey is None:
            continue
        he = snap.get(hkey); ae = snap.get(akey)
        if he is None or ae is None:
            continue
        out_edge.append(he - ae); out_hg.append(hg[k]); out_ag.append(ag[k])
    return pd.DataFrame({'edge': out_edge, 'hg': out_hg, 'ag': out_ag})


def fitted_model(verbose: bool = False) -> EuroGoalModel | None:
    if 'model' in _CACHE:
        return _CACHE['model']
    tr = build_training(verbose=verbose)
    if len(tr) < 2000:
        _CACHE['model'] = None
        return None
    m = EuroGoalModel().fit(tr['edge'], tr['hg'], tr['ag'])
    if verbose:
        print(f'  fitted on {len(tr)} domestic matches with global Elo')
    _CACHE['model'] = m
    return m


def predict_1x2(home: str, away: str, neutral: bool = False):
    """[p_home, p_draw, p_away] for any two European clubs, or None."""
    ratings = current_ratings()
    if ratings is None:
        return None
    eh, ea = elo_of(home, ratings), elo_of(away, ratings)
    if eh is None or ea is None:
        return None
    model = fitted_model()
    if model is None:
        return None
    p = model.market_probs(eh, ea, neutral=neutral)
    return np.array([p['p_home'], p['p_draw'], p['p_away']])


# =============================================================================
# CLI
# =============================================================================
def cmd_update(_a):
    print('Refreshing clubelo snapshots…')
    today = pd.Timestamp.today().strftime('%Y-%m-%d')
    ok = _fetch_snapshot(today) is not None
    print(f'  current snapshot {today}: {"ok" if ok else "FAILED"}')
    dates = ensure_history(verbose=True)
    print(f'  {len(dates)}/{HISTORY_MONTHS} monthly snapshots present')


def cmd_ratings(a):
    r = current_ratings()
    if r is None:
        sys.exit('No ratings available (offline and no cache).')
    r = r.sort_values('Elo', ascending=False).head(a.top)
    print(f"\n  {'#':>3}  {'Club':<24} {'Cty':<4} {'Elo':>7}")
    print('  ' + '-' * 44)
    for k, (_, row) in enumerate(r.iterrows(), 1):
        print(f"  {k:>3}  {row['Club']:<24} {row['Country']:<4} "
              f"{row['Elo']:>7.0f}")


def cmd_predict(a):
    ratings = current_ratings()
    for side in (a.home, a.away):
        if resolve(side, ratings) is None:
            close = [c for c in (ratings['Club'] if ratings is not None else [])
                     if _fold_translit(side)[:4] in _fold_translit(c)]
            sys.exit(f"Unknown club '{side}'. Close: {close[:6]}")
    hn, an = resolve(a.home, ratings), resolve(a.away, ratings)
    eh, ea = elo_of(a.home, ratings), elo_of(a.away, ratings)
    m = fitted_model(verbose=True)
    if m is None:
        sys.exit('Could not fit the goal model (snapshots missing?).')
    p = m.market_probs(eh, ea, neutral=a.neutral)
    venue = 'neutral' if a.neutral else f'{hn} at home'
    print(f"\n  {hn} ({eh:.0f}) vs {an} ({ea:.0f})  [{venue}]")
    print(f"  xG:   {p['lambda_home']:.2f} - {p['lambda_away']:.2f}")
    print(f"  1X2:  {hn} {p['p_home']:5.1%} | Draw {p['p_draw']:5.1%} "
          f"| {an} {p['p_away']:5.1%}")
    print(f"  O2.5: {p['p_over25']:5.1%}   BTTS: {p['p_btts']:5.1%}")


def cmd_backtest(_a):
    """Sanity: fit on the older half of the joined matches, score the newer."""
    tr = build_training(verbose=True)
    if len(tr) < 4000:
        sys.exit(f'Only {len(tr)} joined matches — not enough.')
    mid = len(tr) // 2
    m = EuroGoalModel().fit(tr['edge'][:mid], tr['hg'][:mid], tr['ag'][:mid])
    P = []
    for e in tr['edge'][mid:]:
        p = m.market_probs(1500 + e, 1500)      # only the difference matters
        P.append([p['p_home'], p['p_draw'], p['p_away']])
    P = np.array(P)
    gd = (tr['hg'][mid:] - tr['ag'][mid:]).to_numpy()
    y = np.where(gd > 0, 0, np.where(gd == 0, 1, 2))
    ll = -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None)).mean()
    acc = (P.argmax(1) == y).mean()
    top2 = np.mean([y[i] in np.argsort(P[i])[-2:] for i in range(len(y))])
    print(f"\n  {len(y)} held-out matches: logloss={ll:.4f} acc={acc:.1%} "
          f"top2={top2:.1%}  (uniform = 1.099)")


def main():
    ap = argparse.ArgumentParser(description='European club predictor (clubelo)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('update')
    pr = sub.add_parser('ratings'); pr.add_argument('--top', type=int, default=30)
    pp = sub.add_parser('predict')
    pp.add_argument('--home', required=True); pp.add_argument('--away', required=True)
    pp.add_argument('--neutral', action='store_true')
    sub.add_parser('backtest')
    a = ap.parse_args()
    {'update': cmd_update, 'ratings': cmd_ratings,
     'predict': cmd_predict, 'backtest': cmd_backtest}[a.cmd](a)


if __name__ == '__main__':
    main()
