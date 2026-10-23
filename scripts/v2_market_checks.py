"""
Market checks (2026-10-23): can the models add anything to the odds the app
sees live?
=========================================================================
The live feed gives EARLY prices (football-data collects them a few days
before kickoff, the same kind as AvgH/B365H in the season files). The
"odds win" rule was set against closing prices, so these checks redo it
against the early ones, and test the idea of learning from the gaps between
the goals model, the 1X2 model and the price.

Protocol as everywhere in v2: weights are fitted on the tuning windows
(2022-23) and judged on the test windows (2024-26), with a weekly block
bootstrap. Out-of-sample 1X2 predictions come from
reports/v2/runs/sel_ens_mkt__{tune,test}.pkl; the goals model is refitted
per window (about 45 s in all) and cached in reports/v2/runs/goals_oos.pkl.

    python -m scripts.v2_market_checks
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from sklearn.linear_model import LogisticRegression

from scripts import market
from scripts import v2_goals as vg
from scripts import v2_train as v2

GOALS_OOS = os.path.join(v2.RUN_DIR, 'goals_oos.pkl')
PRE = ['t_avg_pre_H', 't_avg_pre_D', 't_avg_pre_A']
CLOSE = ['t_avg_close_H', 't_avg_close_D', 't_avg_close_A']


# ----------------------------------------------------------------- helpers
def goals_oos(force=False) -> pd.DataFrame:
    if os.path.exists(GOALS_OOS) and not force:
        return pd.read_pickle(GOALS_OOS)
    recipe = vg.tuned_recipe()
    frame = v2.load_cache(recipe['features'])
    rows = []
    for s0 in v2.TUNE_STARTS + v2.TEST_STARTS:
        t0 = pd.Timestamp(s0)
        test = frame[(frame.Date >= t0) & (frame.Date < t0 + pd.Timedelta(days=v2.TEST_DAYS))]
        mk = vg.grid_markets(vg.fit_window(recipe, frame, s0).predict_grid(test))
        rows.append(pd.DataFrame({'match_id': test.match_id.to_numpy(),
                                  **{k: np.asarray(v) for k, v in mk.items()}}))
        print(f'  goals model fitted as of {s0}', flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_pickle(GOALS_OOS)
    return out


def load(part: str, cache: pd.DataFrame, goals: pd.DataFrame) -> pd.DataFrame:
    r = pd.read_pickle(os.path.join(v2.RUN_DIR, f'sel_ens_mkt__{part}.pkl')).set_index('match_id')
    j = r[['Date', 'league', 'FTR', 'pH', 'pD', 'pA']]
    j = j.join(goals.set_index('match_id')[['pH', 'pD', 'pA', 'ou25']].add_prefix('g_'), how='inner')
    j = j.join(cache[PRE + CLOSE + ['Avg>2.5', 'Avg<2.5', 'FTHG', 'FTAG']], how='inner')
    o = pd.to_numeric(j['Avg>2.5'], errors='coerce')
    u = pd.to_numeric(j['Avg<2.5'], errors='coerce')
    j['m_ou25'] = np.where((o > 1) & (u > 1), (1 / o) / (1 / o + 1 / u), np.nan)
    j['g_ou25'] = j['g_ou25'].clip(1e-4, 1 - 1e-4)
    j = j[np.isfinite(j[PRE + CLOSE + ['m_ou25']].to_numpy(float)).all(axis=1)]
    j['y'] = j.FTR.map({'H': 0, 'D': 1, 'A': 2})
    return j


def probs(j, cols):
    p = j[cols].to_numpy(float)
    return p / p.sum(axis=1, keepdims=True)


def _log(p):
    return np.log(np.clip(p, 1e-9, 1))


def softmax(z):
    z = np.exp(z - z.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def ll(p, y):
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))


def ll_bin(p, y):
    return -(y * _log(p) + (1 - y) * _log(1 - p))


def logit(p):
    return np.log(p / (1 - p))


def boot(d, dates, B=2000, seed=0):
    """Mean of d with a 95% weekly block-bootstrap interval."""
    wk = pd.to_datetime(pd.Series(dates)).dt.to_period('W').astype(str).to_numpy()
    u, inv = np.unique(wk, return_inverse=True)
    s, n = np.bincount(inv, d), np.bincount(inv)
    idx = np.random.default_rng(seed).integers(0, len(u), (B, len(u)))
    m = s[idx].sum(1) / n[idx].sum(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def line(name, loss, base, dates):
    d, lo, hi = boot(base - loss, dates)
    print(f'  {name:46} {loss.mean():.5f}   gain {d:+.5f} [{lo:+.5f}, {hi:+.5f}]')


# ----------------------------------------------------------------- checks
def check_1x2(tu, te):
    ytu, yte = tu.y.to_numpy(), te.y.to_numpy()
    Q = lambda j: probs(j, PRE)
    M = lambda j: j[['pH', 'pD', 'pA']].to_numpy(float)
    G = lambda j: j[['g_pH', 'g_pD', 'g_pA']].to_numpy(float)
    base = ll(Q(te), yte)
    print('\n1X2 on test 2024-26 (log-loss; gain = better than the early price)')
    line('early price (what the app uses)', base, base, te.Date)
    line('closing price (not available live)', ll(probs(te, CLOSE), yte), base, te.Date)
    line('1X2 model alone', ll(M(te), yte), base, te.Date)
    line('goals model 1X2 alone', ll(G(te), yte), base, te.Date)

    def pool(j, w):
        return softmax((1 - w[0] - w[1]) * _log(Q(j)) + w[0] * _log(M(j)) + w[1] * _log(G(j)))
    w = minimize(lambda w: ll(pool(tu, w), ytu).mean(), [.1, .1], method='Nelder-Mead').x
    line(f'pool early/1X2/goals {1 - w.sum():.2f}/{w[0]:.2f}/{w[1]:.2f}', ll(pool(te, w), yte), base, te.Date)

    def gaps(j):
        q, m, g = _log(Q(j)), _log(M(j)), _log(G(j))
        return np.c_[q[:, 0] - q[:, 1], q[:, 2] - q[:, 1], m - q, g - q, g - m,
                     logit(j.g_ou25) - logit(j.m_ou25), j.m_ou25]
    for C in (0.01, 0.1, 1.0):
        lr = LogisticRegression(C=C, max_iter=2000).fit(gaps(tu), ytu)
        line(f'gap-learner (logistic, C={C})', ll(lr.predict_proba(gaps(te)), yte), base, te.Date)


def check_line_movement(te):
    M = te[['pH', 'pD', 'pA']].to_numpy(float)
    pre, close = probs(te, PRE), probs(te, CLOSE)
    print('\nLine movement on test: share of the (model - early) gap the price closes by kickoff')
    for k, s in enumerate('HDA'):
        gap, mov = M[:, k] - pre[:, k], close[:, k] - pre[:, k]
        print(f'  {s}: slope {np.polyfit(gap, mov, 1)[0]:+.3f}, correlation {np.corrcoef(gap, mov)[0, 1]:+.3f}')


def check_draws(tu, te):
    def bias(j):
        return (j.y == 1).mean() - probs(j, PRE)[:, 1].mean()
    print(f'\nDraws minus early-price draw chance: 2022-23 {bias(tu) * 100:+.2f} pp, '
          f'2024-26 {bias(te) * 100:+.2f} pp')
    qd = probs(te, PRE)[:, 1]
    for name, gap in (('goals-model O/U minus price O/U', te.g_ou25 - te.m_ou25),
                      ('goals-model draw minus price draw', te.g_pD - qd)):
        t = pd.DataFrame({'gap': gap.to_numpy(), 'pred': qd, 'act': (te.y == 1).to_numpy()})
        t = t.groupby(pd.qcut(t.gap, 5), observed=True).agg(n=('gap', 'size'), pred=('pred', 'mean'), act=('act', 'mean'))
        print(f'  draws by fifths of ({name}): actual minus price, pp (±2 s.e.)')
        for _, r in t.iterrows():
            print(f'    {(r.act - r.pred) * 100:+.1f} ±{200 * np.sqrt(r.pred * (1 - r.pred) / r.n):.1f}  (n {int(r.n)})')


def check_ou(tu, te):
    yo = ((te.FTHG + te.FTAG) >= 3).to_numpy(int)
    yt = ((tu.FTHG + tu.FTAG) >= 3).to_numpy(int)

    def pool(j, w):
        return 1 / (1 + np.exp(-((1 - w) * logit(j.m_ou25) + w * logit(j.g_ou25))))
    w = minimize_scalar(lambda w: ll_bin(pool(tu, w), yt).mean(), bounds=(0, 1), method='bounded').x
    base = ll_bin(te.m_ou25.to_numpy(), yo)
    app = 1 / (1 + np.exp(-(0.118 * logit(te.g_ou25) + 1.1062 * logit(te.m_ou25))))
    print('\nOver/under 2.5 on test (gain = better than the early O/U price)')
    line('early O/U price', base, base, te.Date)
    line('goals model alone', ll_bin(te.g_ou25.to_numpy(), yo), base, te.Date)
    line(f'pool, goals weight {w:.2f} (fitted on tuning)', ll_bin(pool(te, w).to_numpy(), yo), base, te.Date)
    line('app today: 0.118 goals + 1.106 price', ll_bin(app.to_numpy(), yo), base, te.Date)


def check_sources(cache):
    """Which early price in the live feed is sharpest? Same matches for all."""
    f = cache[(cache.Date >= '2024-01-01') & (cache.Date < '2026-09-29')]
    y = f.FTR.map({'H': 0, 'D': 1, 'A': 2}).to_numpy()
    P = {}
    for s in ('Avg', 'B365', 'BFD', 'BV', 'BW', 'Max', 'BFE', 'AvgC'):
        o = f[[f'{s}H', f'{s}D', f'{s}A']].apply(pd.to_numeric, errors='coerce').to_numpy(float)
        ok = np.isfinite(o).all(1) & (o > 1).all(1)
        p = np.full((len(f), 3), np.nan)
        p[ok] = np.asarray(market.shin(o[ok]))
        P[s] = p
    common = np.all([np.isfinite(p).all(1) for p in P.values()], axis=0)
    base = ll(P['Avg'][common], y[common])
    print(f'\nEarly price sources in the live feed, same {common.sum()} matches (gain = better than Avg)')
    for s, p in P.items():
        line(s + (' (closing, for scale)' if s == 'AvgC' else ''), ll(p[common], y[common]), base, f.Date[common])


def main():
    cache = v2.load_cache('mkttot').set_index('match_id')
    goals = goals_oos()
    tu, te = load('tune', cache, goals), load('test', cache, goals)
    print(f'matches with model, goals model, early and closing price: tuning {len(tu)}, test {len(te)}')
    check_1x2(tu, te)
    check_line_movement(te)
    check_draws(tu, te)
    check_ou(tu, te)
    check_sources(cache.reset_index())


if __name__ == '__main__':
    main()
