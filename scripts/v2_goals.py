#!/usr/bin/env python3
"""
V2 goals model — one scoreline distribution for every goal market
=================================================================
The v2 1X2 model (scripts/v2_train.py) says who wins. This says how many
goals: two Poisson rate models (home goals, away goals) on the same causal
features, plus past over/under prices (FeatureState(totals=True)), with a
Dixon-Coles low-score correction and a goal-level scale fitted on the
calibration window. Every goal market is read off one grid of scorelines,
so they cannot contradict each other (P(over 1.5) >= P(over 2.5) >= P(over
3.5); BTTS agrees with them), and it yields expected goals.

The alternative it has to beat on the tuning windows is 'direct': one
classifier per market, like production v1. Same protocol as the 1X2 work:
choose on the 2022-23 tuning windows, confirm once on 2024-26, then compare
with production on matches it never saw.

    python -m scripts.v2_goals exp          # poisson vs direct, tune + test
    python -m scripts.v2_goals compare      # vs production v1 and the market
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from scripts import v2_train as v2

MARKETS = ('ou15', 'ou25', 'ou35', 'btts')
LINES = {'ou15': 2, 'ou25': 3, 'ou35': 4}          # total goals needed for "over"
G = 10                                              # scoreline grid 0..G each side
RUN_DIR = os.path.join(v2.REPORT_DIR, 'goals_runs')

PARAMS = dict(n_estimators=400, max_depth=4, learning_rate=.03,
              min_child_weight=20, reg_lambda=5.0, subsample=.85,
              colsample_bytree=.85)
POISSON = dict(name='goals_poisson', kind='poisson', features='mkttot',
               half_life=500.0, lookback_days=1460, cal_days=90,
               params=dict(PARAMS))
DIRECT = dict(name='goals_direct', kind='direct', features='mkttot',
              half_life=500.0, lookback_days=1460, cal_days=90,
              params=dict(PARAMS))


def labels(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """1 = over / yes for each market."""
    hg, ag = df.FTHG.to_numpy(float), df.FTAG.to_numpy(float)
    out = {m: (hg + ag >= k).astype(int) for m, k in LINES.items()}
    out['btts'] = ((hg > 0) & (ag > 0)).astype(int)
    return out


# =============================================================================
# SCORELINE GRID
# =============================================================================
def grid(lh: np.ndarray, la: np.ndarray, rho: float) -> np.ndarray:
    """(n, G+1, G+1) scoreline probabilities: independent Poissons with the
    Dixon-Coles adjustment on 0-0, 1-0, 0-1 and 1-1, renormalised."""
    from scipy.stats import poisson
    k = np.arange(G + 1)
    ph = poisson.pmf(k[None, :], lh[:, None])
    pa = poisson.pmf(k[None, :], la[:, None])
    g = ph[:, :, None] * pa[:, None, :]
    g[:, 0, 0] *= np.maximum(1 - lh * la * rho, 1e-9)
    g[:, 0, 1] *= np.maximum(1 + lh * rho, 1e-9)
    g[:, 1, 0] *= np.maximum(1 + la * rho, 1e-9)
    g[:, 1, 1] *= np.maximum(1 - rho, 1e-9)
    return g / g.sum(axis=(1, 2), keepdims=True)


_I, _J = np.meshgrid(np.arange(G + 1), np.arange(G + 1), indexing='ij')


def grid_markets(g: np.ndarray) -> dict[str, np.ndarray]:
    """P(over / yes) per market, plus 1X2 and expected goals, from a grid."""
    tot = _I + _J
    out = {m: g[:, tot >= k].sum(axis=1) for m, k in LINES.items()}
    out['btts'] = g[:, (_I > 0) & (_J > 0)].sum(axis=1)
    out['pH'] = g[:, _I > _J].sum(axis=1)
    out['pD'] = g[:, _I == _J].sum(axis=1)
    out['pA'] = g[:, _I < _J].sum(axis=1)
    out['xg_h'] = (g * _I).sum(axis=(1, 2))
    out['xg_a'] = (g * _J).sum(axis=(1, 2))
    return out


def tilt_grid(g: np.ndarray, p1x2=None, ou25=None, btts=None, iters: int = 80,
              tol: float = 1e-8) -> np.ndarray:
    """Re-weight scoreline grids so their margins match given probabilities.

    g      (n, G+1, G+1) grids, home goals on the first axis
    p1x2   (n, 3) target [home, draw, away], or None
    ou25   (n,)   target P(3 or more goals), or None
    btts   (n,)   target P(both teams score), or None

    Iterative proportional fitting. Every target is a partition of the grid
    (the three results; 3+ goals or not; both score or not). A pass scales the
    cells of each region to its target mass, one partition after another,
    until all margins agree. Cells keep their relative weights inside a
    region, so the likeliest scoreline is the likeliest one that is consistent
    with the other predictions.
    """
    n = g.shape[0]
    gf = g.reshape(n, -1).astype(float)
    gf = gf / gf.sum(axis=1, keepdims=True)
    tot = (_I + _J).ravel()
    both = ((_I > 0) & (_J > 0)).ravel()
    parts = []                                    # each: [(mask, target (n,)), ...]

    def two_way(mask, p):
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        return [(mask, p), (~mask, 1 - p)]

    if p1x2 is not None:
        p = np.clip(np.asarray(p1x2, float), 1e-6, None)
        p = p / p.sum(axis=1, keepdims=True)
        parts.append([((_I > _J).ravel(), p[:, 0]), ((_I == _J).ravel(), p[:, 1]),
                      ((_I < _J).ravel(), p[:, 2])])
    if ou25 is not None:
        parts.append(two_way(tot >= 3, ou25))
    if btts is not None:
        parts.append(two_way(both, btts))
    for _ in range(iters if len(parts) > 1 else 1):
        worst = 0.0
        for regions in parts:
            for mask, target in regions:
                cur = gf[:, mask].sum(axis=1)
                worst = max(worst, float(np.max(np.abs(cur - target))))
                gf[:, mask] *= (target / np.clip(cur, 1e-12, None))[:, None]
        if worst < tol:
            break
    return (gf / gf.sum(axis=1, keepdims=True)).reshape(g.shape)


# =============================================================================
# MODEL
# =============================================================================
def _xgb(kind: str, params: dict):
    import xgboost as xgb
    common = dict(tree_method='hist', n_jobs=8, random_state=42, **params)
    if kind == 'poisson':
        return xgb.XGBRegressor(objective='count:poisson', **common)
    return xgb.XGBClassifier(objective='binary:logistic', **common)


def _logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


class GoalsModel:
    """Fitted as of one date; the backtest scores exactly what a bundle serves."""

    def __init__(self, recipe, columns, models, calib, asof):
        self.recipe, self.columns = recipe, list(columns)
        self.models, self.calib, self.asof = models, calib, str(asof)

    def _X(self, df):
        return v2.matrix(df, self.recipe).reindex(columns=self.columns, fill_value=0.0)

    def predict_grid(self, df: pd.DataFrame) -> np.ndarray:
        """(n, G+1, G+1): P(home goals = i, away goals = j) for each fixture."""
        assert self.recipe['kind'] == 'poisson'
        X = self._X(df)
        s, rho = self.calib['scale'], self.calib['rho']
        lh = np.clip(self.models['home'].predict(X) * s, .05, 8)
        la = np.clip(self.models['away'].predict(X) * s, .05, 8)
        return grid(lh, la, rho)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """One row per fixture: P(over 1.5/2.5/3.5), P(BTTS), and for the
        poisson kind also 1X2 and expected goals."""
        X = self._X(df)
        if self.recipe['kind'] == 'poisson':
            return pd.DataFrame(grid_markets(self.predict_grid(df)), index=df.index)
        out = {}
        for m in MARKETS:
            p = self.models[m].predict_proba(X)[:, 1]
            out[m] = 1 / (1 + np.exp(-_logit(p) / self.calib[m]))
        return pd.DataFrame(out, index=df.index)


def _fit_poisson_calib(models, X, hg, ag):
    """Goal-level scale and Dixon-Coles rho by scoreline likelihood."""
    from scipy.optimize import minimize
    lh0, la0 = models['home'].predict(X), models['away'].predict(X)
    hi, ai = np.minimum(hg, G).astype(int), np.minimum(ag, G).astype(int)

    def nll(x):
        s, rho = x
        g = grid(np.clip(lh0 * s, .05, 8), np.clip(la0 * s, .05, 8), rho)
        return -np.log(np.clip(g[np.arange(len(hi)), hi, ai], 1e-12, None)).mean()
    r = minimize(nll, x0=[1.0, 0.0], bounds=[(.8, 1.2), (-.25, .25)], method='L-BFGS-B')
    return {'scale': float(r.x[0]), 'rho': float(r.x[1])}


def _fit_temperature_binary(p, y):
    from scipy.optimize import minimize_scalar
    z = _logit(p)

    def nll(t):
        q = 1 / (1 + np.exp(-z / t))
        return -np.mean(y * np.log(np.clip(q, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - q, 1e-12, 1)))
    return float(minimize_scalar(nll, bounds=(.5, 3.0), method='bounded').x)


def fit_window(recipe: dict, frame: pd.DataFrame, asof) -> GoalsModel:
    fit, cal, _ = v2.split(frame, asof, test_days=1, cal_days=recipe['cal_days'],
                           lookback_days=recipe['lookback_days'])
    X, Xc = v2.matrix(fit, recipe), v2.matrix(cal, recipe)
    w = v2.decay_weights(fit.Date, recipe.get('half_life', 0))
    if recipe['kind'] == 'poisson':
        models = {side: _xgb('poisson', recipe['params']).fit(X, fit[col].to_numpy(float), sample_weight=w)
                  for side, col in (('home', 'FTHG'), ('away', 'FTAG'))}
        calib = _fit_poisson_calib(models, Xc, cal.FTHG.to_numpy(), cal.FTAG.to_numpy())
    else:
        yf, yc = labels(fit), labels(cal)
        models, calib = {}, {}
        for m in MARKETS:
            models[m] = _xgb('direct', recipe['params']).fit(X, yf[m], sample_weight=w)
            calib[m] = _fit_temperature_binary(models[m].predict_proba(Xc)[:, 1], yc[m])
    return GoalsModel(recipe, X.columns, models, calib, asof)


# =============================================================================
# BACKTEST
# =============================================================================
def run(recipe: dict, frame: pd.DataFrame, starts, tag: str, verbose=False) -> pd.DataFrame:
    path = os.path.join(RUN_DIR, f"{recipe['name']}__{tag}.pkl")
    if os.path.isfile(path):
        return pd.read_pickle(path)
    rows = []
    for s in starts:
        t0 = time.time()
        t_start = pd.Timestamp(s)
        test = frame[(frame.Date >= t_start) & (frame.Date < t_start + pd.Timedelta(days=v2.TEST_DAYS))]
        if test.empty:
            continue
        model = fit_window(recipe, frame, s)
        pred = model.predict(test)
        out = test[['match_id', 'Date', 'league', 'FTHG', 'FTAG']].copy()
        out[[f'p_{c}' for c in pred.columns]] = pred.to_numpy()
        out['fold'] = s
        rows.append(out)
        if verbose:
            print(f"    {s} {recipe['name']} calib={model.calib} ({time.time()-t0:.0f}s)")
    res = pd.concat(rows, ignore_index=True)
    os.makedirs(RUN_DIR, exist_ok=True)
    res.to_pickle(path)
    return res


def ll_binary(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired_markets(a: pd.DataFrame, b: pd.DataFrame, pa='p_', pb='p_') -> dict:
    """Per market: mean(loss_a - loss_b) with weekly block bootstrap
    (positive => b better)."""
    j = a.merge(b, on='match_id', suffixes=('_a', '_b'))
    y = labels(j.rename(columns={'FTHG_a': 'FTHG', 'FTAG_a': 'FTAG'}))
    wk = j['Date_a'].dt.strftime('%G-%V').to_numpy()
    out = {}
    for m in MARKETS:
        d = ll_binary(j[f'{pa}{m}_a'], y[m]) - ll_binary(j[f'{pb}{m}_b'], y[m])
        g = pd.Series(d.to_numpy()).groupby(wk).agg(['sum', 'count'])
        s, c = g['sum'].to_numpy(), g['count'].to_numpy()
        idx = np.random.default_rng(0).integers(0, len(s), size=(2000, len(s)))
        boots = s[idx].sum(1) / c[idx].sum(1)
        out[m] = (float(d.mean()), *np.percentile(boots, [2.5, 97.5]).tolist())
    return out


def summary(res: pd.DataFrame) -> dict:
    y = labels(res)
    return {m: float(ll_binary(res[f'p_{m}'].to_numpy(), y[m]).mean()) for m in MARKETS}


def experiment():
    frame = v2.load_cache('mkttot')
    for tag, starts in (('tune', v2.TUNE_STARTS), ('test', v2.TEST_STARTS)):
        rp = run(POISSON, frame, starts, tag, verbose=True)
        rd = run(DIRECT, frame, starts, tag, verbose=True)
        sp, sd = summary(rp), summary(rd)
        pr = paired_markets(rd, rp)
        print(f"\n  [{tag}] log-loss per market (lower is better); gain = poisson better than direct")
        print(f"  {'market':<6} {'poisson':>9} {'direct':>9} {'gain':>9}  95% weekly CI")
        for m in MARKETS:
            d, lo, hi = pr[m]
            print(f"  {m:<6} {sp[m]:>9.5f} {sd[m]:>9.5f} {d:>+9.5f}  [{lo:+.5f}, {hi:+.5f}]"
                  f"{'  *' if lo > 0 or hi < 0 else ''}")
    # coherence: direct models can contradict each other, the grid cannot
    bad = (rd['p_ou15'] < rd['p_ou25']) | (rd['p_ou25'] < rd['p_ou35'])
    print(f"\n  direct models contradicting themselves (P(o1.5) < P(o2.5) or "
          f"P(o2.5) < P(o3.5)): {bad.mean():.2%} of test matches; poisson: 0 by construction")


# =============================================================================
# HYPERPARAMETER SEARCH  (tuning windows only)
# =============================================================================
def _trial_recipe(trial) -> dict:
    r = json.loads(json.dumps(POISSON))
    r['name'] = f'goals_trial_{trial.number}'
    r['half_life'] = trial.suggest_categorical('half_life', [0.0, 250.0, 500.0, 1000.0, 2000.0])
    r['lookback_days'] = trial.suggest_categorical('lookback_days', [1095, 1460, 2190, 3650])
    r['params'] = dict(
        n_estimators=trial.suggest_int('n_estimators', 150, 1500, log=True),
        max_depth=trial.suggest_int('max_depth', 2, 7),
        learning_rate=trial.suggest_float('learning_rate', .005, .1, log=True),
        min_child_weight=trial.suggest_float('min_child_weight', 3, 300, log=True),
        reg_lambda=trial.suggest_float('reg_lambda', .1, 100, log=True),
        reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
        subsample=trial.suggest_float('subsample', .5, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', .3, 1.0),
    )
    return r


def tune(n_trials=60):
    import optuna
    frame = v2.load_cache('mkttot')
    frame = frame[frame.Date < v2.TUNE_END]              # hard wall

    def objective(trial):
        r = _trial_recipe(trial)
        losses = []
        for k, s in enumerate(v2.TUNE_STARTS):
            t = pd.Timestamp(s)
            test = frame[(frame.Date >= t) & (frame.Date < t + pd.Timedelta(days=v2.TEST_DAYS))]
            p, y = fit_window(r, frame, s).predict(test), labels(test)
            losses.append(np.mean([ll_binary(p[m].to_numpy(), y[m]).mean() for m in MARKETS]))
            trial.report(float(np.mean(losses)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(losses))

    study = optuna.create_study(
        study_name='v2_goals_poisson_mkttot', storage=v2.OPTUNA_DB, load_if_exists=True,
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1))
    if not study.trials:
        study.enqueue_trial({**PARAMS, 'reg_alpha': 1e-3, 'half_life': 500.0,
                             'lookback_days': 1460})
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    print(f"  best goals: {study.best_value:.5f} (trial {study.best_trial.number}, "
          f"baseline {study.trials[0].value:.5f})")
    print(json.dumps(study.best_params, indent=2))
    return study


def tuned_recipe() -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    bp = dict(optuna.load_study(study_name='v2_goals_poisson_mkttot',
                                storage=v2.OPTUNA_DB).best_params)
    r = json.loads(json.dumps(POISSON))
    r['name'] = 'goals_poisson_tuned'
    r['half_life'] = bp.pop('half_life')
    r['lookback_days'] = bp.pop('lookback_days')
    r['params'] = bp
    return r


def confirm():
    """Tuned recipe vs the untuned poisson on the tuning windows (where it was
    chosen) and once on the test windows."""
    frame = v2.load_cache('mkttot')
    r = tuned_recipe()
    for tag, starts in (('tune', v2.TUNE_STARTS), ('test', v2.TEST_STARTS)):
        base, tuned = run(POISSON, frame, starts, tag), run(r, frame, starts, tag)
        pr = paired_markets(base, tuned)
        st = summary(tuned)
        print(f"  [{tag}] tuned vs untuned: " + '  '.join(
            f"{m} {st[m]:.5f} ({d:+.5f} [{lo:+.5f},{hi:+.5f}])" for m, (d, lo, hi) in pr.items()))
    with open(os.path.join(v2.REPORT_DIR, 'selected_goals_recipe.json'), 'w') as f:
        json.dump(r, f, indent=2)


# =============================================================================
# HEAD-TO-HEAD vs PRODUCTION v1 (and the over/under market)
# =============================================================================
def compare(recipe: dict | None = None):
    """v2 goals vs the production per-market models on matches after each
    production model file was written; v2 trained as of the earliest cutoff."""
    from scripts import config, data_loader, utils
    from scripts import blend as blend_mod
    from scripts import v2_compare as vc
    recipe = recipe or tuned_recipe()
    data = data_loader.load_processed_data()
    data['Date'] = pd.to_datetime(data['Date']).dt.normalize()
    rows = []
    for league in utils.get_available_leagues():
        feats = config.get_features_for_league(league)
        for m in MARKETS:
            days = [vc._file_day(utils._model_path(m, league, 1, 'current', e))
                    for e in config.MODEL_TYPES]
            days = [d for d in days if d is not None]
            if not days:
                continue
            ldf = data[(data.league == league) & (data.Date > max(days))].dropna(subset=feats)
            if ldf.empty:
                continue
            p = blend_mod.ensemble_proba_market(m, league, ldf[feats])
            if p is None:
                continue
            rows.append(ldf[vc.KEY].assign(market=m, prod=np.asarray(p)[:, 1],
                                           cutoff=max(days)))
    prod = pd.concat(rows, ignore_index=True)
    frame = v2.load_cache(recipe['features'])
    asof = prod.cutoff.min() + pd.Timedelta(days=1)
    model = fit_window(recipe, frame, asof)
    test = frame[frame.Date >= asof]
    pv = model.predict(test)
    pv = pd.concat([test[vc.KEY + ['FTHG', 'FTAG']].reset_index(drop=True),
                    pv.reset_index(drop=True)], axis=1)
    # the over/under 2.5 market where priced before kick-off (rich leagues)
    o = test['Avg>2.5'].apply(pd.to_numeric, errors='coerce').to_numpy() if 'Avg>2.5' in test else None
    u = test['Avg<2.5'].apply(pd.to_numeric, errors='coerce').to_numpy() if 'Avg<2.5' in test else None
    pv['mkt_ou25'] = np.where((o > 1) & (u > 1), (1 / o) / (1 / o + 1 / u), np.nan) if o is not None else np.nan
    pv['Date'] = pd.to_datetime(pv['Date']).dt.normalize()
    j = prod.merge(pv, on=vc.KEY)
    print(f"  v2 goals ({recipe['name']}) vs production v1, matches after each "
          f"production model was written (v2 trained as of {asof.date()})\n")
    print(f"  {'market':<6} {'n':>6} {'v1':>8} {'v2':>8} {'v2 gain':>9}  95% weekly CI")
    out = {}
    for m in MARKETS:
        jm = j[j.market == m]
        y = labels(jm)[m]
        l1, l2 = ll_binary(jm['prod'].to_numpy(), y), ll_binary(jm[m].to_numpy(), y)
        d, lo, hi, wk = vc._paired(l1 - l2, jm['Date'])
        out[m] = dict(n=int(len(jm)), v1=float(l1.mean()), v2=float(l2.mean()),
                      gain=float(d), ci=[float(lo), float(hi)])
        print(f"  {m:<6} {len(jm):>6} {l1.mean():>8.4f} {l2.mean():>8.4f} {d:>+9.4f}  "
              f"[{lo:+.4f}, {hi:+.4f}]{'  *' if lo > 0 else ''}")
    jm = j[(j.market == 'ou25') & j.mkt_ou25.notna()]
    if len(jm) > 100:
        y = labels(jm)['ou25']
        lm = ll_binary(jm['mkt_ou25'].to_numpy(), y)
        l1, l2 = ll_binary(jm['prod'].to_numpy(), y), ll_binary(jm['ou25'].to_numpy(), y)
        d2 = vc._paired(lm - l2, jm['Date'])
        print(f"\n  over/under 2.5 with pre-match odds ({len(jm):,}): market {lm.mean():.4f} | "
              f"v2 {l2.mean():.4f} (vs market {d2[0]:+.4f} [{d2[1]:+.4f}, {d2[2]:+.4f}]) | "
              f"v1 {l1.mean():.4f}")
        out['ou25_vs_market'] = dict(n=int(len(jm)), market=float(lm.mean()),
                                     v2=float(l2.mean()), v1=float(l1.mean()))
    with open(os.path.join(v2.REPORT_DIR, 'compare_goals.json'), 'w') as f:
        json.dump(out, f, indent=2)
    return out


# =============================================================================
# CORRECT SCORE
# =============================================================================
def _actual_cells(df: pd.DataFrame):
    return (np.minimum(df.FTHG.to_numpy(int), G), np.minimum(df.FTAG.to_numpy(int), G))


def league_prior(fit: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """No-information baseline: each league's own score frequencies in the
    fitting period (half a count of smoothing per cell)."""
    out = np.zeros((len(test), G + 1, G + 1))
    cnt = {}
    hi, ai = _actual_cells(fit)
    for lg, idx in fit.reset_index(drop=True).groupby('league').groups.items():
        c = np.full((G + 1, G + 1), 0.5)
        np.add.at(c, (hi[idx], ai[idx]), 1)
        cnt[lg] = c / c.sum()
    allc = np.full((G + 1, G + 1), 0.5)
    np.add.at(allc, (hi, ai), 1)
    allc /= allc.sum()
    for k, lg in enumerate(test.league.to_numpy()):
        out[k] = cnt.get(lg, allc)
    return out


def score_experiment(recipe: dict | None = None, starts=None) -> pd.DataFrame:
    """Which scoreline model finds the real score best? Same windows and
    protocol as the 1X2 work: fit before each window, predict the window."""
    from scripts import v2_compare as vc
    recipe = recipe or tuned_recipe()
    starts = starts or v2.TEST_STARTS
    frame = v2.load_cache(recipe['features'])
    best1x2 = pd.read_pickle(os.path.join(v2.RUN_DIR, 'sel_ens_mkt__test.pkl')).set_index('match_id')
    rows = []
    for s0 in starts:
        t0 = pd.Timestamp(s0)
        test = frame[(frame.Date >= t0) & (frame.Date < t0 + pd.Timedelta(days=v2.TEST_DAYS))]
        test = test[test.match_id.isin(best1x2.index)]
        if test.empty:
            continue
        fit, cal, _ = v2.split(frame, s0, test_days=1, cal_days=recipe['cal_days'],
                               lookback_days=recipe['lookback_days'])
        model = fit_window(recipe, frame, s0)
        raw = model.predict_grid(test)
        b = best1x2.loc[test.match_id]
        p1 = np.c_[b.pH, b.pD, b.pA]                       # [home, draw, away]
        mk = test[['t_avg_pre_H', 't_avg_pre_D', 't_avg_pre_A']].to_numpy(float)
        o = pd.to_numeric(test.get('Avg>2.5'), errors='coerce').to_numpy(float)
        u = pd.to_numeric(test.get('Avg<2.5'), errors='coerce').to_numpy(float)
        ou_mk = np.where((o > 1) & (u > 1), (1 / o) / (1 / o + 1 / np.where(u > 1, u, np.nan)), np.nan)
        has_mk = np.isfinite(mk).all(axis=1)
        hi, ai = _actual_cells(test)
        variants = {
            'baseline (league frequencies)': league_prior(pd.concat([fit, cal]), test),
            'model grid': raw,
            'grid tilted to the 1X2 model': tilt_grid(raw, p1=None) if False else tilt_grid(raw, p1x2=p1),
        }
        m = grid_markets(raw)
        variants['grid tilted to 1X2, keeping O/U + BTTS'] = tilt_grid(raw, p1, m['ou25'], m['btts'])
        mk_grid = raw.copy()
        if has_mk.any():
            ou_t = np.where(np.isfinite(ou_mk), ou_mk, m['ou25'])
            mk_grid[has_mk] = tilt_grid(raw[has_mk], mk[has_mk], ou_t[has_mk], m['btts'][has_mk])
        variants['grid tilted to the market (odds, O/U)'] = mk_grid
        for name, G_ in variants.items():
            pa = np.clip(G_[np.arange(len(test)), hi, ai], 1e-12, None)
            flat = G_.reshape(len(test), -1)
            top = np.argsort(-flat, axis=1)[:, :3]
            act = hi * (G + 1) + ai
            mm = grid_markets(G_)
            yb = ((test.FTHG > 0) & (test.FTAG > 0)).to_numpy(int)
            yo = ((test.FTHG + test.FTAG) >= 3).to_numpy(int)
            rows.append(pd.DataFrame({
                'match_id': test.match_id.to_numpy(), 'Date': test.Date.to_numpy(), 'variant': name,
                'll': -np.log(pa), 'top1': (top[:, 0] == act).astype(float),
                'top3': (top == act[:, None]).any(axis=1).astype(float),
                'll_btts': ll_binary(mm['btts'], yb), 'll_ou25': ll_binary(mm['ou25'], yo),
                'has_market': has_mk, 'pmax': flat.max(axis=1)}))
    res = pd.concat(rows, ignore_index=True)
    res['Date'] = pd.to_datetime(res['Date'])
    return res


def print_score_report(res: pd.DataFrame) -> dict:
    from scripts import v2_compare as vc
    names = list(dict.fromkeys(res.variant))
    out = {}
    for label, sub in (('all matches', res), ('matches with bookmaker odds', res[res.has_market])):
        n = sub.match_id.nunique()
        print(f"\n  {label}: {n:,} matches")
        print(f"  {'scoreline model':<42} {'log-loss':>9} {'top-1':>7} {'top-3':>7}  gain vs the plain model grid (95% weekly CI)")
        base = sub[sub.variant == names[0]].set_index('match_id')
        raw = sub[sub.variant == names[1]].set_index('match_id')
        for nm in names:
            v = sub[sub.variant == nm].set_index('match_id')
            d = base.ll - v.ll.reindex(base.index)
            mean, lo, hi, _ = vc._paired(d.to_numpy(), base.Date)
            d2 = raw.ll - v.ll.reindex(raw.index)
            m2, lo2, hi2, _ = vc._paired(d2.to_numpy(), raw.Date)
            extra = '' if nm in names[:2] else f"{m2:+.4f} [{lo2:+.4f}, {hi2:+.4f}]{'  *' if lo2 > 0 else ''}"
            print(f"  {nm:<42} {v.ll.mean():>9.4f} {v.top1.mean():>7.1%} {v.top3.mean():>7.1%}  {extra}")
            if nm not in names[:1]:
                print(f"      same grid read as BTTS: {v.ll_btts.mean():.5f}   over/under 2.5: {v.ll_ou25.mean():.5f}")
            out.setdefault(label, {})[nm] = dict(n=int(n), ll=float(v.ll.mean()), top1=float(v.top1.mean()),
                                                 ll_btts=float(v.ll_btts.mean()), ll_ou25=float(v.ll_ou25.mean()),
                                                 top3=float(v.top3.mean()), gain_vs_baseline=float(mean),
                                                 ci_vs_baseline=[float(lo), float(hi)], gain_vs_grid=float(m2),
                                                 ci_vs_grid=[float(lo2), float(hi2)])
    return out


def main():
    ap = argparse.ArgumentParser(description='V2 goals model')
    ap.add_argument('command', choices=['exp', 'tune', 'compare', 'confirm', 'score'])
    ap.add_argument('--trials', type=int, default=60)
    a = ap.parse_args()
    if a.command == 'exp':
        experiment()
    elif a.command == 'tune':
        tune(a.trials)
    elif a.command == 'compare':
        compare()
    elif a.command == 'confirm':
        confirm()
    elif a.command == 'score':
        res = score_experiment()
        rep = print_score_report(res)
        with open(os.path.join(v2.REPORT_DIR, 'score_eval.json'), 'w') as f:
            json.dump(rep, f, indent=2)


if __name__ == '__main__':
    main()
