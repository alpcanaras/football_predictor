#!/usr/bin/env python3
"""
V2 training pipeline — tuned, distilled, all-market models
===========================================================
Built on the causal feature engine (scripts/research_features.py): one
day-batched FeatureState produces features for historical replay AND upcoming
fixtures, so training and live prediction cannot drift apart the way the
production features.py / utils.py pair did.

Nothing here touches production. Models go to models/v2/, reports to
reports/v2/, and the app keeps loading models/tier1/ until a v2 bundle is
promoted by hand after its report.

Evaluation protocol (fixed before any experiment is run):
  * six out-of-sample test windows of 180 days starting 2024-01-01 … 2026-07-01
    (the same windows as the research branch, so numbers are comparable);
  * per window: fit on a trailing lookback ending 90 days before the test,
    calibrate on those 90 days, score the test window;
  * paired comparisons with weekly block-bootstrap intervals, because
    same-weekend matches are not independent;
  * hyperparameter search only ever sees data before 2024-01-01 (TUNE_END),
    and the chosen settings are frozen before the test windows are scored.

    python -m scripts.v2_train cache            # replay features once
    python -m scripts.v2_train baseline         # reproduce the research recipe
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from scripts import config, market
from scripts import research_features as rf

ROOT = config.PROJECT_ROOT
REPORT_DIR = os.path.join(ROOT, 'reports', 'v2')
MODEL_DIR = os.path.join(ROOT, 'models', 'v2')
CACHE = os.path.join(REPORT_DIR, 'features.pkl')
# Feature sets: 'v1' is the research engine as-is; 'mkt' adds ratings fitted
# to past pre-match prices (research_features.FeatureState(market=True)).
FEATURE_SETS = ('v1', 'mkt')


def cache_path(features: str = 'v1') -> str:
    if features not in FEATURE_SETS:
        raise ValueError(f'unknown feature set {features!r}')
    return CACHE if features == 'v1' else CACHE.replace('.pkl', f'_{features}.pkl')

TEST_STARTS = ['2024-01-01', '2024-07-01', '2025-01-01',
               '2025-07-01', '2026-01-01', '2026-07-01']
TEST_DAYS = 180
TUNE_END = pd.Timestamp('2024-01-01')      # tuning never sees anything later
# Validation windows for SELECTING settings — all end before TUNE_END, so the
# test windows above are only ever used to confirm a choice already made.
TUNE_STARTS = ['2022-01-01', '2022-07-01', '2023-01-01', '2023-07-01']

# Market "teachers" for distillation, best first. Closing prices are fine as
# TRAINING TARGETS — like the final score, they are facts about a finished
# match. They are never features, and never used for a match being predicted.
TEACHERS = {
    'pinnacle_close': ('PSCA', 'PSCD', 'PSCH'),
    'avg_close': ('AvgCA', 'AvgCD', 'AvgCH'),
    'avg_pre': ('AvgA', 'AvgD', 'AvgH'),
    'b365_pre': ('B365A', 'B365D', 'B365H'),
}


# =============================================================================
# FEATURE CACHE
# =============================================================================
def _teacher_probs(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Shin-de-vigged [A, D, H] per teacher source; NaN where unavailable."""
    out = {}
    for name, cols in TEACHERS.items():
        if not set(cols).issubset(df.columns):
            out[name] = np.full((len(df), 3), np.nan)
            continue
        odds = df[list(cols)].apply(pd.to_numeric, errors='coerce').to_numpy(float)
        out[name] = market.shin(odds)
    return out


def build_cache(verbose=True, features: str = 'v1') -> pd.DataFrame:
    path = cache_path(features)
    t0 = time.time()
    raw, manifest = rf.load_raw()
    if verbose:
        print(f"  loaded {len(raw):,} matches in {time.time()-t0:.0f}s; replaying…")
    t1 = time.time()
    feats, _ = rf.replay(raw, rf.FeatureState(market=features == 'mkt'))
    if verbose:
        print(f"  replayed in {time.time()-t1:.0f}s")
    feats = feats.sort_values(['Date'] + rf.KEY[:1] + rf.KEY[2:],
                              kind='stable').reset_index(drop=True)
    for name, p in _teacher_probs(feats).items():
        for k, c in enumerate('ADH'):
            feats[f't_{name}_{c}'] = p[:, k]
    os.makedirs(REPORT_DIR, exist_ok=True)
    feats.to_pickle(path)
    meta = {
        'built_at': pd.Timestamp.now().isoformat(timespec='seconds'),
        'feature_set': features,
        'feature_version': rf.version(features == 'mkt'),
        'n_matches': int(len(feats)),
        'date_min': str(feats.Date.min().date()),
        'date_max': str(feats.Date.max().date()),
        'raw_files_sha256': hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        'n_raw_files': len(manifest),
    }
    with open(path + '.json', 'w') as f:
        json.dump(meta, f, indent=2)
    if verbose:
        print(f"  cached {len(feats):,} rows x {feats.shape[1]} cols -> {path}")
    return feats


def load_cache(features: str = 'v1') -> pd.DataFrame:
    path = cache_path(features)
    if not os.path.isfile(path):
        return build_cache(features=features)
    return pd.read_pickle(path)


# =============================================================================
# EVALUATION HARNESS
# =============================================================================
RUN_DIR = os.path.join(REPORT_DIR, 'runs')

# The research branch's fixed football recipe, reproduced exactly so every v2
# number is measured against a baseline verified here rather than assumed.
BASELINE = dict(
    name='baseline_research', engine='xgb', target='hard', teacher=None,
    alpha=0.0, half_life=500.0, lookback_days=1460, cal_days=90,
    calib='temperature', features='v1',
    params=dict(n_estimators=250, max_depth=3, learning_rate=.035,
                min_child_weight=30, reg_lambda=10, reg_alpha=.1,
                subsample=.85, colsample_bytree=.85),
)


def labels(df: pd.DataFrame) -> np.ndarray:
    return df.FTR.map({'A': 0, 'D': 1, 'H': 2}).to_numpy(int)


def matrix(df: pd.DataFrame, recipe: dict | None = None) -> pd.DataFrame:
    """Causal features + a one-hot per registered league (fixed schema).
    A missing feature column raises: never silently zero-fill a feature."""
    market = (recipe or {}).get('features', 'v1') == 'mkt'
    x = df[rf.feature_columns(market=market)].astype(float).reset_index(drop=True)
    for lg in sorted(config.LEAGUE_REGISTRY):
        x['league_' + lg] = (df.league.to_numpy() == lg).astype(float)
    return x


def decay_weights(dates: pd.Series, half_life: float) -> np.ndarray:
    if not half_life or half_life <= 0:
        return np.ones(len(dates))
    w = .5 ** ((dates.max() - dates).dt.days.to_numpy(float) / half_life)
    return w / w.mean()


def split(frame, test_start, test_days=TEST_DAYS, cal_days=90,
          lookback_days=1460):
    """fit | cal | test, strictly ordered in time with whole days kept together."""
    t0 = pd.Timestamp(test_start)
    cal0 = t0 - pd.Timedelta(days=cal_days)
    lo = t0 - pd.Timedelta(days=lookback_days)
    fit = frame[(frame.Date >= lo) & (frame.Date < cal0)]
    cal = frame[(frame.Date >= cal0) & (frame.Date < t0)]
    test = frame[(frame.Date >= t0) & (frame.Date < t0 + pd.Timedelta(days=test_days))]
    assert fit.Date.max() < cal.Date.min() <= cal.Date.max() < t0
    return fit, cal, test


def soft_targets(df: pd.DataFrame, recipe: dict) -> np.ndarray:
    """(n, 3) training targets in class order [A, D, H].

    hard: one-hot result. With a teacher, alpha*market + (1-alpha)*one-hot —
    matches the teacher never priced fall back to the pure result.
    """
    y = labels(df)
    onehot = np.eye(3)[y]
    if recipe.get('target') == 'hard' or not recipe.get('teacher'):
        return onehot
    t = recipe['teacher']
    q = df[[f't_{t}_A', f't_{t}_D', f't_{t}_H']].to_numpy(float)
    ok = np.isfinite(q).all(axis=1)
    a = float(recipe.get('alpha', 1.0))
    out = onehot.copy()
    out[ok] = a * q[ok] + (1 - a) * onehot[ok]
    return out


def expand(X: pd.DataFrame, T: np.ndarray, w: np.ndarray):
    """Soft-label cross-entropy as weighted hard labels: each match becomes
    three rows (one per outcome) weighted by its target probability. Exactly
    equivalent in the loss, and works with any multi-class learner."""
    n = len(X)
    X3 = pd.concat([X, X, X], ignore_index=True)
    y3 = np.repeat(np.arange(3)[None, :], n, axis=0).T.ravel()
    w3 = (T.T * w[None, :]).ravel()
    keep = w3 > 0
    return X3[keep].reset_index(drop=True), y3[keep], w3[keep]


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_base(fit: pd.DataFrame, recipe: dict):
    X = matrix(fit, recipe)
    w = decay_weights(fit.Date, recipe.get('half_life', 0))
    X3, y3, w3 = expand(X, soft_targets(fit, recipe), w)
    p = dict(recipe['params'])
    if recipe['engine'] == 'xgb':
        import xgboost as xgb
        m = xgb.XGBClassifier(objective='multi:softprob', num_class=3,
                              tree_method='hist', n_jobs=8,
                              random_state=42, **p)
        m.fit(X3, y3, sample_weight=w3)
    elif recipe['engine'] == 'lgbm':
        import lightgbm as lgb
        m = lgb.LGBMClassifier(objective='multiclass', num_class=3, n_jobs=8,
                               random_state=42, verbose=-1, **p)
        m.fit(X3, y3, sample_weight=w3)
    else:
        raise ValueError(recipe['engine'])
    return m


def fit_temperature(p: np.ndarray, y: np.ndarray) -> float:
    from scipy.optimize import minimize_scalar
    lp = np.log(np.clip(p, 1e-12, 1))
    def nll(t):
        q = _softmax(lp / t)
        return -np.log(np.clip(q[np.arange(len(y)), y], 1e-12, None)).mean()
    r = minimize_scalar(nll, bounds=(.5, 3.0), method='bounded')
    return float(min([.5, 3.0, r.x], key=nll))


def apply_temperature(p: np.ndarray, t: float) -> np.ndarray:
    return _softmax(np.log(np.clip(p, 1e-12, 1)) / t)


class V2Model:
    """What a recipe produces for one as-of date: one or more fitted members
    averaged, then temperature-scaled. The backtest scores exactly this
    object, and the bundle saves exactly this object, so what was measured
    is what gets served."""

    def __init__(self, models, temperature, columns, recipe, asof):
        self.models, self.temperature = models, float(temperature)
        self.columns = [list(c) for c in columns]          # one list per member
        self.recipe, self.asof = recipe, str(asof)
        self.members = recipe.get('members') or [recipe]

    def raw_proba(self, df: pd.DataFrame) -> np.ndarray:
        # reindex: a league registered after training gets all-zero one-hots
        # (features themselves are never filled: matrix() raises if missing)
        return np.mean([m.predict_proba(matrix(df, r).reindex(columns=c, fill_value=0.0))
                        for m, r, c in zip(self.models, self.members, self.columns)],
                       axis=0)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Class order [A, D, H]."""
        return apply_temperature(self.raw_proba(df), self.temperature)


def fit_window(recipe: dict, frame: pd.DataFrame, asof) -> tuple[V2Model, dict]:
    """Fit a recipe using only matches strictly before `asof`: each member on
    its own lookback up to the calibration window, then one temperature on
    the shared calibration window (the last cal_days before `asof`)."""
    members = recipe.get('members') or [recipe]
    cal_days = recipe.get('cal_days', 90)
    models, pcs, cols, info = [], [], [], {}
    for r in members:
        fit, cal, _ = split(frame, asof, test_days=1, cal_days=cal_days,
                            lookback_days=r['lookback_days'])
        m = fit_base(fit, r)
        pcs.append(m.predict_proba(matrix(cal, r)))
        cols.append(matrix(cal.head(1), r).columns)
        if r.get('refit'):
            # the calibration window only chose T; refit the trees on it too
            # so the served model has seen the most recent days
            m = fit_base(pd.concat([fit, cal]), r)
        models.append(m)
        info = {'n_fit': int(len(fit)), 'n_cal': int(len(cal))}
    temp = 1.0
    if recipe.get('calib', 'temperature') == 'temperature':
        temp = fit_temperature(np.mean(pcs, axis=0), labels(cal))
    return V2Model(models, temp, cols, recipe, asof), info


def run(recipe: dict, frame: pd.DataFrame | None = None,
        starts=TEST_STARTS, save=True, verbose=True,
        test_days=TEST_DAYS) -> pd.DataFrame:
    """Score a recipe on every outer window; returns one row per test match."""
    feats = {r.get('features', 'v1') for r in recipe.get('members') or [recipe]}
    if frame is None:
        frame = load_cache('mkt' if 'mkt' in feats else 'v1')
    rows = []
    t_all = time.time()
    for s in starts:
        t0 = time.time()
        t_start = pd.Timestamp(s)
        test = frame[(frame.Date >= t_start)
                     & (frame.Date < t_start + pd.Timedelta(days=test_days))]
        if test.empty:
            continue
        model, info = fit_window(recipe, frame, s)
        p = model.predict_proba(test)
        temp = model.temperature
        out = test[['match_id', 'Date', 'league', 'FTR', 'qA', 'qD', 'qH']].copy()
        out['fold'] = s
        out['y'] = labels(test)
        out[['pA', 'pD', 'pH']] = p
        out['temperature'] = temp
        rows.append(out)
        if verbose:
            ll = -np.log(np.clip(p[np.arange(len(test)), labels(test)], 1e-12, None)).mean()
            print(f"    {s}  fit={info['n_fit']:>6,} cal={info['n_cal']:>5,} test={len(test):>5,}"
                  f"  T={temp:.3f}  ll={ll:.4f}  ({time.time()-t0:.0f}s)")
    res = pd.concat(rows, ignore_index=True)
    if save:
        os.makedirs(RUN_DIR, exist_ok=True)
        res.to_pickle(os.path.join(RUN_DIR, f"{recipe['name']}.pkl"))
        with open(os.path.join(RUN_DIR, f"{recipe['name']}.json"), 'w') as f:
            json.dump({**recipe, 'seconds': round(time.time() - t_all)}, f,
                      indent=2, default=str)
    return res


def ll_vec(res: pd.DataFrame) -> np.ndarray:
    P = res[['pA', 'pD', 'pH']].to_numpy()
    return -np.log(np.clip(P[np.arange(len(res)), res['y'].to_numpy()], 1e-12, None))


def market_subset(res: pd.DataFrame) -> pd.Series:
    return np.isfinite(res[['qA', 'qD', 'qH']]).all(axis=1) & res[['qA', 'qD', 'qH']].gt(0).all(axis=1)


def summary(res: pd.DataFrame) -> dict:
    ll = ll_vec(res)
    P = res[['pA', 'pD', 'pH']].to_numpy(); y = res['y'].to_numpy()
    m = market_subset(res).to_numpy()
    top2 = (np.argsort(P, axis=1)[:, 1:] == y[:, None]).any(axis=1)
    return {'n': int(len(res)), 'll_all': float(ll.mean()),
            'll_market_subset': float(ll[m].mean()),
            'n_market_subset': int(m.sum()), 'acc': float((P.argmax(1) == y).mean()),
            'top2': float(top2.mean())}


def paired(a: pd.DataFrame, b: pd.DataFrame, subset=None, B=2000, seed=0):
    """mean(loss_a - loss_b) over shared matches, weekly block bootstrap.
    Positive => b is better."""
    j = a[['match_id', 'Date']].assign(la=ll_vec(a)).merge(
        b[['match_id']].assign(lb=ll_vec(b)), on='match_id')
    if subset is not None:
        j = j[j.match_id.isin(subset)]
    d = (j.la - j.lb).to_numpy()
    wk = pd.Series(d).groupby(j.Date.dt.strftime('%G-%V').to_numpy()).agg(['sum', 'count'])
    s, c = wk['sum'].to_numpy(), wk['count'].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(B, len(s)))
    boots = s[idx].sum(1) / c[idx].sum(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {'n': len(d), 'weeks': len(s), 'diff': d.mean(), 'lo': lo, 'hi': hi}


# =============================================================================
# EXPERIMENTS
# =============================================================================
def variant(**changes) -> dict:
    """The baseline with some settings changed (params merged, not replaced)."""
    r = json.loads(json.dumps(BASELINE))
    params = changes.pop('params', None)
    r.update(changes)
    if params:
        r['params'].update(params)
    return r


# chosen on the tuning windows by the 'distill' experiment (a=0.5 best there)
BEST_DISTILL = dict(target='soft', teacher='avg_close', alpha=.5)


EXPERIMENTS = {
    # E1: learn the market's function of football features, not raw results
    'distill': [
        variant(name='d_avgclose_a50', target='soft', teacher='avg_close', alpha=.5),
        variant(name='d_avgclose_a80', target='soft', teacher='avg_close', alpha=.8),
        variant(name='d_avgclose_a100', target='soft', teacher='avg_close', alpha=1.0),
        variant(name='d_pinclose_a100', target='soft', teacher='pinnacle_close', alpha=1.0),
        variant(name='d_avgpre_a100', target='soft', teacher='avg_pre', alpha=1.0),
    ],
    # E2: ratings fitted to past pre-match prices, on top of distillation
    'mktfeat': [
        variant(name='d_avgclose_a50', **BEST_DISTILL),
        variant(name='mkt_d_avgclose_a50', features='mkt', **BEST_DISTILL),
        variant(name='mkt_hard', features='mkt'),
    ],
}


def run_experiment(name: str):
    """Run each recipe on the tuning windows (to choose) and the test windows
    (to confirm), always paired against the baseline on the same matches."""
    recipes = EXPERIMENTS[name]
    need_mkt = any(m.get('features') == 'mkt' for r in recipes
                   for m in (r.get('members') or [r]))
    # the mkt cache is the v1 cache plus extra columns, so it serves both
    frame = load_cache('mkt' if need_mkt else 'v1')
    out = {}
    for tag, starts in (('tune', TUNE_STARTS), ('test', TEST_STARTS)):
        base_name = f"{BASELINE['name']}__{tag}"
        base_path = os.path.join(RUN_DIR, base_name + '.pkl')
        if os.path.isfile(base_path):
            base = pd.read_pickle(base_path)
        else:
            base = run({**BASELINE, 'name': base_name}, frame, starts, verbose=False)
        bs = summary(base)
        print(f"\n  [{tag}] {len(starts)} windows, baseline ll_all={bs['ll_all']:.5f} "
              f"top2={bs['top2']:.4f}")
        print(f"  {'recipe':<22} {'ll_all':>9} {'top2':>7} {'gain vs base':>13} "
              f"{'95% weekly CI':>24}")
        for r in recipes:
            rr = {**r, 'name': f"{r['name']}__{tag}"}
            path = os.path.join(RUN_DIR, rr['name'] + '.pkl')
            res = pd.read_pickle(path) if os.path.isfile(path) \
                else run(rr, frame, starts, verbose=False)
            sm, pr = summary(res), paired(base, res)
            out[(r['name'], tag)] = (sm, pr)
            flag = '  *' if pr['lo'] > 0 else ''
            print(f"  {r['name']:<22} {sm['ll_all']:>9.5f} {sm['top2']:>7.4f} "
                  f"{pr['diff']:>+13.5f} [{pr['lo']:+.5f}, {pr['hi']:+.5f}]{flag}")
    print("\n  gain > 0 means better than the baseline; * = interval excludes zero")
    return out


# =============================================================================
# HYPERPARAMETER SEARCH  (tuning windows only — never the test windows)
# =============================================================================
OPTUNA_DB = 'sqlite:///' + os.path.join(REPORT_DIR, 'optuna.db')


def _trial_recipe(trial, engine: str, features: str = 'v1') -> dict:
    r = variant(name=f'trial_{engine}_{trial.number}', engine=engine,
                features=features, **BEST_DISTILL)
    r['alpha'] = trial.suggest_float('alpha', .3, 1.0)
    r['half_life'] = trial.suggest_categorical(
        'half_life', [0.0, 250.0, 500.0, 1000.0, 2000.0])
    r['lookback_days'] = trial.suggest_categorical(
        'lookback_days', [1095, 1460, 2190, 3650])
    if engine == 'xgb':
        r['params'] = dict(
            n_estimators=trial.suggest_int('n_estimators', 150, 1500, log=True),
            max_depth=trial.suggest_int('max_depth', 2, 7),
            learning_rate=trial.suggest_float('learning_rate', .005, .1, log=True),
            min_child_weight=trial.suggest_float('min_child_weight', 3, 300, log=True),
            reg_lambda=trial.suggest_float('reg_lambda', .1, 100, log=True),
            reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
            gamma=trial.suggest_float('gamma', 0, 5),
            subsample=trial.suggest_float('subsample', .5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', .3, 1.0),
        )
    else:  # lgbm — bagging enabled explicitly (subsample needs subsample_freq)
        r['params'] = dict(
            n_estimators=trial.suggest_int('n_estimators', 150, 1500, log=True),
            num_leaves=trial.suggest_int('num_leaves', 4, 64, log=True),
            max_depth=trial.suggest_int('max_depth', 2, 8),
            learning_rate=trial.suggest_float('learning_rate', .005, .1, log=True),
            min_child_samples=trial.suggest_int('min_child_samples', 20, 800, log=True),
            reg_lambda=trial.suggest_float('reg_lambda', 1e-3, 100, log=True),
            reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
            subsample=trial.suggest_float('subsample', .5, 1.0),
            subsample_freq=1,
            colsample_bytree=trial.suggest_float('colsample_bytree', .3, 1.0),
        )
    return r


def tune(engine='xgb', n_trials=100, features='v1'):
    import optuna
    frame = load_cache(features)
    frame = frame[frame.Date < TUNE_END]          # hard wall: no test data

    def objective(trial):
        r = _trial_recipe(trial, engine, features)
        losses = []
        for k, s in enumerate(TUNE_STARTS):
            res = run(r, frame, [s], save=False, verbose=False)
            losses.append(ll_vec(res).mean())
            trial.report(float(np.mean(losses)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(losses))

    study = optuna.create_study(
        study_name=f'v2_{engine}_distill' + ('' if features == 'v1' else f'_{features}'),
        storage=OPTUNA_DB,
        load_if_exists=True, direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1))
    if len(study.trials) == 0 and engine == 'xgb':
        # start from the known reference points: the distilled baseline, and
        # the best settings a finished v1-feature search found (if any)
        b = BASELINE['params']
        study.enqueue_trial({**b, 'gamma': 0.0, 'alpha': .5,
                             'half_life': 500.0, 'lookback_days': 1460})
        if features != 'v1':
            try:
                prior = optuna.load_study(study_name='v2_xgb_distill',
                                          storage=OPTUNA_DB)
                study.enqueue_trial(prior.best_params)
            except Exception:
                pass
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True,
                   show_progress_bar=False)
    best = study.best_trial
    print(f"  best {engine}: ll_all={best.value:.5f} (trial {best.number})")
    print(json.dumps(best.params, indent=2))
    return study


def main():
    ap = argparse.ArgumentParser(description='V2 training pipeline')
    ap.add_argument('command', choices=['cache', 'baseline', 'exp', 'tune'])
    ap.add_argument('name', nargs='?')
    ap.add_argument('--engine', default='xgb', choices=['xgb', 'lgbm'])
    ap.add_argument('--trials', type=int, default=100)
    ap.add_argument('--features', default='v1', choices=FEATURE_SETS)
    a = ap.parse_args()
    if a.command == 'cache':
        build_cache(features=a.features)
    elif a.command == 'tune':
        tune(a.engine, a.trials, a.features)
    elif a.command == 'exp':
        run_experiment(a.name)
    elif a.command == 'baseline':
        res = run(BASELINE)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                          for k, v in summary(res).items()}, indent=2))


if __name__ == '__main__':
    main()
