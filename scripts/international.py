"""
International / World Cup Predictor
====================================
National-team model built on data/global/international.csv (all internationals
since 1872, martj42 dataset) — completely separate from the club pipeline,
because none of the club features (league position, odds columns, shots)
exist for national teams.

Two layers:
  1. Tournament-weighted Elo (eloratings.net scheme): K scaled by competition
     importance, goal-margin multiplier, home advantage only at non-neutral
     venues.
  2. Poisson goal model: expected goals for each side as a function of the
     Elo difference, fitted on recent internationals. Converts ratings into
     score distributions -> 1X2 / O-U / BTTS probabilities.

Usage:
  python scripts/international.py ratings  [--top 30]
  python scripts/international.py predict  --home Brazil --away Morocco [--venue neutral|home]
  python scripts/international.py backtest
  python scripts/international.py wc2026   [--sims 20000]

Refresh the dataset (played results + upcoming WC fixtures) with:
  python scripts/international.py update
"""

import argparse
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTL_FILE = os.path.join(PROJECT_ROOT, 'data', 'global', 'international.csv')
WC_FIXTURES_FILE = os.path.join(PROJECT_ROOT, 'data', 'global', 'wc2026_fixtures.csv')
RESULTS_URL = ('https://raw.githubusercontent.com/martj42/'
               'international_results/master/results.csv')

# =============================================================================
# ELO PARAMETERS (eloratings.net scheme)
# =============================================================================
ELO_INITIAL = 1500.0
ELO_HOME_ADV = 80.0          # added to home rating when venue is not neutral

# K-factor by competition importance
K_WORLD_CUP = 60
K_CONTINENTAL_FINALS = 50
K_QUALIFIERS = 40
K_NATIONS_LEAGUE = 40
K_MINOR_TOURNAMENT = 30
K_FRIENDLY = 20

CONTINENTAL_FINALS = {
    'UEFA Euro', 'Copa América', 'African Cup of Nations',
    'AFC Asian Cup', 'Gold Cup', 'CONCACAF Championship',
    'Oceania Nations Cup', 'Confederations Cup', 'CONMEBOL–UEFA Cup of Champions',
}
MINOR_KEYWORDS = (
    'King', 'Kirin', 'Baltic', 'Nordic', 'Gulf Cup', 'SAFF', 'CECAFA',
    'COSAFA', 'EAFF', 'AFF', 'Island Games', 'Merdeka', 'Nehru',
)


def tournament_k(tournament: str) -> float:
    t = str(tournament)
    if t == 'FIFA World Cup':
        return K_WORLD_CUP
    if t in CONTINENTAL_FINALS:
        return K_CONTINENTAL_FINALS
    if 'qualification' in t:
        return K_QUALIFIERS
    if 'Nations League' in t:
        return K_NATIONS_LEAGUE
    if t == 'Friendly':
        return K_FRIENDLY
    if any(k in t for k in MINOR_KEYWORDS):
        return K_MINOR_TOURNAMENT
    return K_MINOR_TOURNAMENT


def margin_multiplier(goal_diff: int) -> float:
    """eloratings.net goal-difference multiplier."""
    d = abs(goal_diff)
    if d <= 1:
        return 1.0
    if d == 2:
        return 1.5
    return (11.0 + d) / 8.0


def expected_score(rating_diff: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-rating_diff / 400.0))


# =============================================================================
# DATA + ELO PASS
# =============================================================================
def load_results() -> pd.DataFrame:
    df = pd.read_csv(INTL_FILE, parse_dates=['date'])
    df = df.dropna(subset=['home_score', 'away_score'])
    df = df.sort_values('date').reset_index(drop=True)
    df['home_score'] = df['home_score'].astype(int)
    df['away_score'] = df['away_score'].astype(int)
    return df


def run_elo(df: pd.DataFrame):
    """Single chronological Elo pass.

    Returns (ratings, history_df) where history_df carries the pre-match
    ratings of both sides for every row of df (needed for fitting the goal
    model and for leak-free backtests).
    """
    ratings: dict[str, float] = defaultdict(lambda: ELO_INITIAL)
    n = len(df)
    pre_home = np.empty(n)
    pre_away = np.empty(n)

    homes = df['home_team'].to_numpy()
    aways = df['away_team'].to_numpy()
    hgs = df['home_score'].to_numpy()
    ags = df['away_score'].to_numpy()
    neutrals = df['neutral'].astype(bool).to_numpy()
    ks = df['tournament'].map(tournament_k).to_numpy()

    for i in range(n):
        h, a = homes[i], aways[i]
        rh, ra = ratings[h], ratings[a]
        pre_home[i] = rh
        pre_away[i] = ra

        diff = (rh + (0.0 if neutrals[i] else ELO_HOME_ADV)) - ra
        exp_h = expected_score(diff)
        gd = int(hgs[i]) - int(ags[i])
        actual = 1.0 if gd > 0 else (0.0 if gd < 0 else 0.5)
        delta = ks[i] * margin_multiplier(gd) * (actual - exp_h)
        ratings[h] = rh + delta
        ratings[a] = ra - delta

    hist = df.copy()
    hist['elo_home_pre'] = pre_home
    hist['elo_away_pre'] = pre_away
    return dict(ratings), hist


# =============================================================================
# GOAL MODEL (Poisson score grid + direct 1X2 head)
# =============================================================================
# Two heads, blended for 1X2:
#   * Poisson-with-Dixon-Coles-tau -> the score grid (O/U, BTTS, xG) and a
#     structurally-constrained 1X2.
#   * Softmax regression straight to 1X2 -> no Poisson bottleneck, free to
#     shape the draw probability.
# Validated on 11,076 rolling out-of-sample internationals (2015+, refit
# yearly, leak-free): plain Poisson 0.8778 log-loss / 82.8% top-2, +tau 0.8770,
# 60/40 blend 0.8743 / 83.7% top-2 (paired t = +4.35 vs tau alone). Top-2 is
# what a Toto "double" covers, hence the emphasis.
MIX_MULTINOM = 0.60          # weight on the softmax head in the 1X2 log-pool


class GoalModel:
    """Elo edge -> goal rates and 1X2 probabilities for national teams."""

    MAX_GOALS = 10

    def __init__(self):
        self.coef_ = None       # Poisson coefficients
        self.rho_ = 0.0         # Dixon-Coles low-score correction
        self.W_ = None          # softmax 1X2 weights

    @staticmethod
    def _design(elo_edge, is_home_nonneutral, is_friendly):
        n = len(np.atleast_1d(elo_edge))
        return np.column_stack([
            np.ones(n),
            np.asarray(elo_edge, dtype=float) / 400.0,
            np.asarray(is_home_nonneutral, dtype=float) * np.ones(n),
            np.asarray(is_friendly, dtype=float) * np.ones(n),
        ])

    @staticmethod
    def _design_1x2(elo_edge, is_home_nonneutral, is_friendly):
        """Richer design for the direct head: a saturating and a symmetric
        term let it price big mismatches and toss-ups differently."""
        e = np.asarray(elo_edge, dtype=float) / 400.0
        n = len(np.atleast_1d(e))
        return np.column_stack([
            np.ones(n), e, np.tanh(e),
            np.asarray(is_home_nonneutral, dtype=float) * np.ones(n),
            np.asarray(is_friendly, dtype=float) * np.ones(n),
            np.abs(e),
        ])

    def fit(self, hist: pd.DataFrame, since: str = '2010-01-01'):
        d = hist[hist['date'] >= since]
        friendly = (d['tournament'] == 'Friendly').to_numpy(dtype=float)
        neutral = d['neutral'].astype(bool).to_numpy()
        edge = (d['elo_home_pre'] - d['elo_away_pre']).to_numpy()
        hg = d['home_score'].to_numpy(float)
        ag = d['away_score'].to_numpy(float)

        X = np.vstack([
            self._design(edge, ~neutral, friendly),
            self._design(-edge, np.zeros(len(d), dtype=bool), friendly),
        ])
        y = np.concatenate([hg, ag])
        self.coef_ = self._fit_poisson_irls(X, y)
        self.rho_ = self._fit_rho(edge, neutral, friendly, hg, ag)

        gd = hg - ag
        outcome = np.where(gd > 0, 0, np.where(gd == 0, 1, 2))
        self.W_ = self._fit_softmax(
            self._design_1x2(edge, ~neutral, friendly), outcome)
        return self

    def _fit_rho(self, edge, neutral, friendly, hg, ag):
        """Profile-fit the Dixon-Coles low-score correction (lambdas fixed)."""
        lam, mu = self._lambda_arrays(edge, neutral, friendly)
        hg = hg.astype(int); ag = ag.astype(int)
        t00 = (hg == 0) & (ag == 0); t01 = (hg == 0) & (ag == 1)
        t10 = (hg == 1) & (ag == 0); t11 = (hg == 1) & (ag == 1)

        def nll(rho):
            tau = np.ones(len(hg))
            tau[t00] = 1.0 - lam[t00] * mu[t00] * rho
            tau[t01] = 1.0 + lam[t01] * rho
            tau[t10] = 1.0 + mu[t10] * rho
            tau[t11] = 1.0 - rho
            return -np.log(np.clip(tau, 1e-9, None)).sum()

        lo, hi = -0.30, 0.30                     # ternary search (nll is convex)
        for _ in range(80):
            m1 = lo + (hi - lo) / 3.0
            m2 = hi - (hi - lo) / 3.0
            if nll(m1) < nll(m2):
                hi = m2
            else:
                lo = m1
        return (lo + hi) / 2.0

    @staticmethod
    def _fit_softmax(X, y, iters=300, l2=1e-4, step=8.0):
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

    @staticmethod
    def _fit_poisson_irls(X, y, n_iter=50, tol=1e-8):
        beta = np.zeros(X.shape[1])
        beta[0] = math.log(max(y.mean(), 1e-3))
        for _ in range(n_iter):
            eta = X @ beta
            mu = np.exp(np.clip(eta, -20, 5))
            W = mu
            z = eta + (y - mu) / np.maximum(mu, 1e-9)
            XtW = X.T * W
            new_beta = np.linalg.solve(XtW @ X, XtW @ z)
            if np.max(np.abs(new_beta - beta)) < tol:
                beta = new_beta
                break
            beta = new_beta
        return beta

    def _lambda_arrays(self, edge, neutral, friendly):
        """Vectorised goal rates for arrays of matches."""
        edge = np.atleast_1d(np.asarray(edge, dtype=float))
        neutral = np.atleast_1d(np.asarray(neutral, dtype=bool)) \
            * np.ones(len(edge), dtype=bool)
        xh = self._design(edge, ~neutral, friendly)
        xa = self._design(-edge, np.zeros(len(edge)), friendly)
        lam = np.exp(np.clip(xh @ self.coef_, -20, 5))
        mu = np.exp(np.clip(xa @ self.coef_, -20, 5))
        return lam, mu

    def lambdas(self, elo_home, elo_away, neutral=True, friendly=False):
        lam, mu = self._lambda_arrays([elo_home - elo_away], neutral, friendly)
        return float(lam[0]), float(mu[0])

    def score_grid(self, lam_h, lam_a):
        """Independent-Poisson grid with the Dixon-Coles low-score correction."""
        g = np.arange(self.MAX_GOALS + 1)
        fact = np.array([math.factorial(int(i)) for i in g])
        ph = np.exp(-lam_h) * lam_h ** g / fact
        pa = np.exp(-lam_a) * lam_a ** g / fact
        grid = np.outer(ph, pa)
        if self.rho_:
            r = self.rho_
            grid[0, 0] *= max(1.0 - lam_h * lam_a * r, 1e-10)
            grid[0, 1] *= max(1.0 + lam_h * r, 1e-10)
            grid[1, 0] *= max(1.0 + lam_a * r, 1e-10)
            grid[1, 1] *= max(1.0 - r, 1e-10)
        return grid / grid.sum()

    def probs_1x2_direct(self, elo_home, elo_away, neutral=True, friendly=False):
        """Softmax head: [p_home, p_draw, p_away]."""
        x = self._design_1x2([elo_home - elo_away], [not neutral], [friendly])
        z = x @ self.W_
        z -= z.max()
        p = np.exp(z).ravel()
        return p / p.sum()

    def market_probs(self, elo_home, elo_away, neutral=True, friendly=False):
        lam_h, lam_a = self.lambdas(elo_home, elo_away, neutral, friendly)
        grid = self.score_grid(lam_h, lam_a)
        i, j = np.indices(grid.shape)
        regions = [i > j, i == j, i < j]
        p_grid = np.array([grid[r].sum() for r in regions])

        # Blend the two 1X2 heads, then tilt the score grid so its own 1X2
        # marginals match the blend — keeps O/U and BTTS coherent with the
        # 1X2 we actually report.
        if self.W_ is not None:
            p_dir = self.probs_1x2_direct(elo_home, elo_away, neutral, friendly)
            lg = (MIX_MULTINOM * np.log(np.clip(p_dir, 1e-12, None))
                  + (1.0 - MIX_MULTINOM) * np.log(np.clip(p_grid, 1e-12, None)))
            lg -= lg.max()
            p_mix = np.exp(lg); p_mix /= p_mix.sum()
            for r, target, base in zip(regions, p_mix, p_grid):
                if base > 1e-12:
                    grid[r] *= target / base
            grid /= grid.sum()
        else:
            p_mix = p_grid

        total = i + j
        return {
            'lambda_home': lam_h,
            'lambda_away': lam_a,
            'p_home': float(p_mix[0]),
            'p_draw': float(p_mix[1]),
            'p_away': float(p_mix[2]),
            'p_over15': grid[total > 1.5].sum(),
            'p_over25': grid[total > 2.5].sum(),
            'p_over35': grid[total > 3.5].sum(),
            'p_btts': grid[(i > 0) & (j > 0)].sum(),
            'grid': grid,
        }


# =============================================================================
# COMMANDS
# =============================================================================
def cmd_update(_args):
    # Use requests (not urllib) — urllib hits SSL: CERTIFICATE_VERIFY_FAILED on
    # macOS's bundled certs; requests uses the system trust store.
    import io
    import requests
    print(f"Downloading {RESULTS_URL} ...")
    resp = requests.get(RESULTS_URL, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    played = df.dropna(subset=['home_score', 'away_score']).copy()
    played[['home_score', 'away_score']] = played[['home_score', 'away_score']].astype(int)
    played.to_csv(INTL_FILE, index=False)
    fix = df[df['home_score'].isna()].drop(columns=['home_score', 'away_score'])
    fix.to_csv(WC_FIXTURES_FILE, index=False)
    print(f"  international.csv     {len(played)} matches through {played['date'].max()}")
    print(f"  wc2026_fixtures.csv   {len(fix)} upcoming fixtures")


def cmd_ratings(args):
    df = load_results()
    ratings, hist = run_elo(df)
    # Only teams active in the last 2 years
    cutoff = df['date'].max() - pd.Timedelta(days=730)
    recent = df[df['date'] >= cutoff]
    active = set(recent['home_team']) | set(recent['away_team'])
    table = sorted(((r, t) for t, r in ratings.items() if t in active), reverse=True)
    print(f"\n  Elo ratings (data through {df['date'].max().date()})\n")
    print(f"  {'#':>3}  {'Team':<24} {'Elo':>7}")
    print(f"  {'-'*3}  {'-'*24} {'-'*7}")
    for idx, (r, t) in enumerate(table[:args.top], 1):
        print(f"  {idx:>3}  {t:<24} {r:>7.0f}")


def _print_prediction(home, away, probs, neutral):
    venue = 'neutral venue' if neutral else f'{home} at home'
    print(f"\n  {home} vs {away}  ({venue})")
    print(f"  xG: {probs['lambda_home']:.2f} - {probs['lambda_away']:.2f}")
    print(f"  1X2 : {home} {probs['p_home']:5.1%} | Draw {probs['p_draw']:5.1%} "
          f"| {away} {probs['p_away']:5.1%}")
    print(f"  O/U : O1.5 {probs['p_over15']:5.1%} | O2.5 {probs['p_over25']:5.1%} "
          f"| O3.5 {probs['p_over35']:5.1%}")
    print(f"  BTTS: {probs['p_btts']:5.1%}")


def cmd_predict(args):
    df = load_results()
    ratings, hist = run_elo(df)
    for team in (args.home, args.away):
        if team not in ratings:
            close = [t for t in ratings if team.lower() in t.lower()]
            sys.exit(f"Unknown team '{team}'. Close matches: {close[:8]}")
    model = GoalModel().fit(hist)
    neutral = args.venue == 'neutral'
    probs = model.market_probs(ratings[args.home], ratings[args.away],
                               neutral=neutral, friendly=args.friendly)
    _print_prediction(args.home, args.away, probs, neutral)
    print(f"\n  Elo: {args.home} {ratings[args.home]:.0f} "
          f"vs {args.away} {ratings[args.away]:.0f}")


def cmd_backtest(_args):
    """Leak-free backtest on WC 2018, WC 2022, Euro 2024.

    Elo pre-match ratings are inherently leak-free (chronological pass).
    The goal model is refit using only matches before each tournament.
    Note: knockout scores in the dataset include extra time, which slightly
    inflates decisive results vs a 90-minute 1X2 market.
    """
    df = load_results()
    _, hist = run_elo(df)

    tournaments = [
        ('World Cup 2018', 'FIFA World Cup', '2018-06-01', '2018-08-01'),
        ('World Cup 2022', 'FIFA World Cup', '2022-11-01', '2023-01-01'),
        ('Euro 2024', 'UEFA Euro', '2024-06-01', '2024-08-01'),
    ]
    print(f"\n  {'Tournament':<16} {'N':>4} {'Acc':>7} {'Top2':>7} "
          f"{'LogLoss':>8} {'Brier':>7}  (uniform LL = 1.099)")
    print('  ' + '-' * 62)

    overall = []
    for name, tourn, start, end in tournaments:
        mask = ((hist['tournament'] == tourn)
                & (hist['date'] >= start) & (hist['date'] < end))
        test = hist[mask]
        train = hist[hist['date'] < start]
        model = GoalModel().fit(train)

        lls, briers, accs, top2s = [], [], [], []
        for _, m in test.iterrows():
            p = model.market_probs(m['elo_home_pre'], m['elo_away_pre'],
                                   neutral=bool(m['neutral']))
            probs = np.array([p['p_home'], p['p_draw'], p['p_away']])
            gd = m['home_score'] - m['away_score']
            actual = 0 if gd > 0 else (1 if gd == 0 else 2)
            onehot = np.zeros(3)
            onehot[actual] = 1.0
            lls.append(-math.log(max(probs[actual], 1e-12)))
            briers.append(np.sum((probs - onehot) ** 2))
            accs.append(int(np.argmax(probs) == actual))
            top2s.append(int(actual in np.argsort(probs)[-2:]))
            overall.append((lls[-1], briers[-1], accs[-1], top2s[-1]))
        print(f"  {name:<16} {len(test):>4} {np.mean(accs):>7.1%} "
              f"{np.mean(top2s):>7.1%} {np.mean(lls):>8.3f} {np.mean(briers):>7.3f}")

    o = np.array(overall)
    print('  ' + '-' * 62)
    print(f"  {'OVERALL':<16} {len(o):>4} {o[:,2].mean():>7.1%} "
          f"{o[:,3].mean():>7.1%} {o[:,0].mean():>8.3f} {o[:,1].mean():>7.3f}")


def _infer_groups(*frames: pd.DataFrame) -> list[set]:
    """Recover the groups from the match graph (each group is a connected
    component of 'has a group match against'). Pass played results as well as
    remaining fixtures so groups stay complete once matches start being
    played and drop out of the fixtures file."""
    adj = defaultdict(set)
    for fr in frames:
        if fr is None or fr.empty:
            continue
        for h, a in zip(fr['home_team'], fr['away_team']):
            adj[h].add(a)
            adj[a].add(h)
    seen, groups = set(), []
    for team in adj:
        if team in seen:
            continue
        comp, stack = set(), [team]
        while stack:
            t = stack.pop()
            if t in comp:
                continue
            comp.add(t)
            stack.extend(adj[t] - comp)
        seen |= comp
        groups.append(comp)
    return groups


def wc_played(df: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """World Cup matches of this edition that are already played."""
    if fixtures.empty:
        return fixtures.iloc[0:0]
    season_start = pd.Timestamp(fixtures['date'].min()) - pd.Timedelta(days=60)
    return df[(df['tournament'] == 'FIFA World Cup')
              & (df['date'] >= season_start)]


def _standings(played: pd.DataFrame, t_idx: dict):
    """Points / goal-difference / goals-for accumulated so far."""
    n = len(t_idx)
    pts = np.zeros(n); gd = np.zeros(n); gf = np.zeros(n)
    for _, m in played.iterrows():
        h, a = t_idx.get(m['home_team']), t_idx.get(m['away_team'])
        if h is None or a is None:
            continue
        hg, ag = int(m['home_score']), int(m['away_score'])
        gd[h] += hg - ag; gd[a] += ag - hg
        gf[h] += hg; gf[a] += ag
        if hg > ag:
            pts[h] += 3
        elif ag > hg:
            pts[a] += 3
        else:
            pts[h] += 1; pts[a] += 1
    return pts, gd, gf


def cmd_wc2026(args):
    df = load_results()
    ratings, hist = run_elo(df)
    model = GoalModel().fit(hist)
    fixtures = pd.read_csv(WC_FIXTURES_FILE, parse_dates=['date'])
    fixtures = fixtures[fixtures['tournament'] == 'FIFA World Cup']

    missing = (set(fixtures['home_team']) | set(fixtures['away_team'])) - set(ratings)
    if missing:
        sys.exit(f"Teams missing from ratings: {missing}")

    # --- per-match predictions ---
    print(f"\n  WORLD CUP 2026 — GROUP-STAGE PREDICTIONS "
          f"(ratings through {df['date'].max().date()})\n")
    print(f"  {'Date':<11} {'Match':<42} {'1':>6} {'X':>6} {'2':>6} "
          f"{'O2.5':>6} {'BTTS':>6}")
    print('  ' + '-' * 82)
    match_probs = []
    for _, m in fixtures.sort_values('date').iterrows():
        neutral = bool(m['neutral'])
        p = model.market_probs(ratings[m['home_team']], ratings[m['away_team']],
                               neutral=neutral)
        match_probs.append(p)
        label = f"{m['home_team']} vs {m['away_team']}"
        print(f"  {str(m['date'].date()):<11} {label:<42} "
              f"{p['p_home']:>6.1%} {p['p_draw']:>6.1%} {p['p_away']:>6.1%} "
              f"{p['p_over25']:>6.1%} {p['p_btts']:>6.1%}")

    # --- Monte-Carlo group simulation ---
    played = wc_played(df, fixtures)
    groups = _infer_groups(fixtures, played)
    sizes = sorted({len(g) for g in groups})
    if any(len(g) > 4 for g in groups):
        print("\n  (Group simulation skipped: the match graph has components "
              "larger than 4 — knockout matches have started, so group "
              "advancement no longer applies.)")
        return
    if any(len(g) < 3 for g in groups):
        print(f"\n  (Group simulation skipped: incomplete groups {sizes} — "
              "not enough of this edition's matches are in the dataset yet.)")
        return

    print(f"\n  GROUP SIMULATION ({args.sims} runs) — top 2 advance, "
          f"plus 8 best third-placed teams")
    if len(played):
        print(f"  Seeded with {len(played)} already-played group matches; "
              f"simulating the {len(fixtures)} remaining.\n")
    else:
        print()
    team_group = {t: gi for gi, g in enumerate(groups) for t in g}
    teams = sorted(team_group)
    t_idx = {t: i for i, t in enumerate(teams)}

    # Sample scorelines from each fixture's corrected score grid (not raw
    # Poisson) so the simulation matches the 1X2/O-U probabilities printed
    # above — the grid carries both the tau correction and the 1X2 blend.
    hidx, aidx, cdfs = [], [], []
    for k, (_, m) in enumerate(fixtures.sort_values('date').iterrows()):
        hidx.append(t_idx[m['home_team']])
        aidx.append(t_idx[m['away_team']])
        cdfs.append(np.cumsum(match_probs[k]['grid'].ravel()))
    hidx = np.array(hidx); aidx = np.array(aidx)
    side = GoalModel.MAX_GOALS + 1
    n_fix = len(hidx)

    rng = np.random.default_rng(42)
    n_teams = len(teams)
    S = int(args.sims)
    group_of = np.array([team_group[t] for t in teams])

    # (S, n_fix) scorelines, drawn in one vectorised pass
    gh = np.empty((S, n_fix), dtype=np.int16)
    ga = np.empty((S, n_fix), dtype=np.int16)
    for k in range(n_fix):
        cell = np.searchsorted(cdfs[k], rng.random(S))
        np.clip(cell, 0, side * side - 1, out=cell)
        gh[:, k] = cell // side
        ga[:, k] = cell % side

    pts0, gd0, gf0 = _standings(played, t_idx)     # results already on the board
    pts = np.tile(pts0, (S, 1)).astype(float)
    gdiff = np.tile(gd0, (S, 1)).astype(float)
    gfor = np.tile(gf0, (S, 1)).astype(float)
    home_pts = np.where(gh > ga, 3, np.where(gh == ga, 1, 0))
    away_pts = np.where(ga > gh, 3, np.where(gh == ga, 1, 0))
    np.add.at(pts, (slice(None), hidx), home_pts)
    np.add.at(pts, (slice(None), aidx), away_pts)
    np.add.at(gdiff, (slice(None), hidx), gh - ga)
    np.add.at(gdiff, (slice(None), aidx), ga - gh)
    np.add.at(gfor, (slice(None), hidx), gh)
    np.add.at(gfor, (slice(None), aidx), ga)

    # rank inside each group: points, GD, goals scored, random tiebreak
    key = pts * 1e6 + gdiff * 1e3 + gfor + rng.random((S, n_teams))
    adv_count = np.zeros(n_teams)
    win_group_count = np.zeros(n_teams)
    thirds_idx = np.empty((S, len(groups)), dtype=int)
    for gi in range(len(groups)):
        members = np.where(group_of == gi)[0]
        order = members[np.argsort(-key[:, members], axis=1)]     # (S, 4)
        np.add.at(win_group_count, order[:, 0], 1)
        np.add.at(adv_count, order[:, 0], 1)
        np.add.at(adv_count, order[:, 1], 1)
        thirds_idx[:, gi] = order[:, 2]
    third_keys = np.take_along_axis(key, thirds_idx, axis=1)
    best8 = np.take_along_axis(thirds_idx, np.argsort(-third_keys, axis=1)[:, :8],
                               axis=1)
    np.add.at(adv_count, best8.ravel(), 1)

    print(f"  {'Team':<24} {'Group':>5} {'Elo':>6} {'Win grp':>8} {'Advance':>8}")
    print('  ' + '-' * 56)
    order = np.argsort(-adv_count)
    for i in order:
        t = teams[i]
        print(f"  {t:<24} {group_of[i]+1:>5} {ratings[t]:>6.0f} "
              f"{win_group_count[i]/args.sims:>8.1%} {adv_count[i]/args.sims:>8.1%}")


def main():
    parser = argparse.ArgumentParser(description='International / World Cup predictor')
    sub = parser.add_subparsers(dest='command', required=True)

    sub.add_parser('update', help='Refresh results + WC fixtures from GitHub')

    p_rat = sub.add_parser('ratings', help='Show current Elo ratings')
    p_rat.add_argument('--top', type=int, default=30)

    p_pred = sub.add_parser('predict', help='Predict a single fixture')
    p_pred.add_argument('--home', required=True)
    p_pred.add_argument('--away', required=True)
    p_pred.add_argument('--venue', choices=['neutral', 'home'], default='neutral')
    p_pred.add_argument('--friendly', action='store_true')

    sub.add_parser('backtest', help='Backtest on WC 2018/2022 + Euro 2024')

    p_wc = sub.add_parser('wc2026', help='Predict all WC 2026 group fixtures')
    p_wc.add_argument('--sims', type=int, default=20000)

    args = parser.parse_args()
    {'update': cmd_update, 'ratings': cmd_ratings, 'predict': cmd_predict,
     'backtest': cmd_backtest, 'wc2026': cmd_wc2026}[args.command](args)


if __name__ == '__main__':
    main()
