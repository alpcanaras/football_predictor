#!/usr/bin/env python3
"""
Training Script V2
==================
Optuna-tuned XGBoost + LightGBM per league per market, with early stopping,
temporal calibration, and checkpoint/resume.

Usage:
    python scripts/train.py                        # Full training (resumable)
    python scripts/train.py --models 1x2 ou25      # Specific markets
    python scripts/train.py --leagues british_pl    # Specific leagues
    python scripts/train.py --fresh                 # Ignore checkpoint, retrain all
    python scripts/train.py --validate-only         # Validation only
"""

import argparse
import json
import warnings
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import optuna
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error

from scripts import config
from scripts import data_loader
from scripts import utils

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# CHECKPOINT
# =============================================================================
def _checkpoint_path(model_set: str = 'current') -> str:
    tier_dir = config.get_tier_dir(tier=1, model_set=model_set)
    return os.path.join(os.path.dirname(tier_dir), '.train_checkpoint.json')


def _load_checkpoint(model_set: str = 'current') -> set:
    path = _checkpoint_path(model_set)
    if os.path.exists(path):
        with open(path) as f:
            return set(json.load(f).get('done', []))
    return set()


def _save_checkpoint(done: set, model_set: str = 'current'):
    path = _checkpoint_path(model_set)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'done': sorted(done)}, f, indent=2)


def _ck(league, market, engine='', sub=''):
    parts = [league, market]
    if engine:
        parts.append(engine)
    if sub:
        parts.append(sub)
    return '|'.join(parts)


def clear_checkpoint(model_set: str = 'current'):
    path = _checkpoint_path(model_set)
    if os.path.exists(path):
        os.remove(path)


# =============================================================================
# SHARED HELPERS
# =============================================================================
def _ts_splits(n: int) -> int:
    mx = max(int(getattr(config, 'CV_MAX_SPLITS', 5)), 2)
    mn = max(int(getattr(config, 'CV_MIN_TEST_SIZE', 120)), 10)
    by_test = max((n // mn) - 1, 2)
    return max(min(mx, by_test, n - 1), 2)


def _time_decay_weights(dates, half_life):
    if half_life is None or half_life <= 0:
        return None
    if dates.empty:
        return None
    age = (dates.max() - dates).dt.days.clip(lower=0).astype(float)
    raw = 0.5 ** (age / float(half_life))
    w = raw.to_numpy(np.float32)
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        return None
    w *= np.float32(len(w) / s)
    return w


def _xgb_base() -> dict:
    return {
        'verbosity': 0,
        'n_jobs': int(getattr(config, 'XGB_N_JOBS', 4)),
        'tree_method': getattr(config, 'XGB_TREE_METHOD', 'hist'),
        'random_state': 42,
        'max_bin': int(getattr(config, 'XGB_MAX_BIN', 256)),
    }


def _cal_split(n: int):
    frac = float(getattr(config, 'CALIBRATION_HOLDOUT_FRACTION', 0.15))
    min_cal = max(int(getattr(config, 'CALIBRATION_MIN_SAMPLES', 120)), 20)
    max_cal = int(getattr(config, 'CALIBRATION_MAX_SAMPLES', 700))
    min_fit = max(int(getattr(config, 'CALIBRATION_MIN_FIT_SAMPLES', 400)), 100)
    cal_sz = max(int(round(n * frac)), min_cal)
    if max_cal > 0:
        cal_sz = min(cal_sz, max_cal)
    if n - cal_sz < min_fit:
        cal_sz = n - min_fit
    if cal_sz < min_cal:
        return None
    fit_end = n - cal_sz
    if fit_end < min_fit:
        return None
    return fit_end, cal_sz


def _pick_cal_method(y_cal, requested):
    req = (requested or 'sigmoid').lower()
    if req in ('sigmoid', 'isotonic'):
        return req
    is_binary = pd.Series(y_cal).nunique(dropna=True) == 2
    iso_min = int(getattr(config, 'CALIBRATION_ISOTONIC_MIN_SAMPLES', 500))
    if is_binary and len(y_cal) >= iso_min:
        return 'isotonic'
    return 'sigmoid'


ES = int(getattr(config, 'EARLY_STOPPING_ROUNDS', 50))
MAX_EST = int(getattr(config, 'MAX_ESTIMATORS', 3000))
N_TRIALS = int(getattr(config, 'OPTUNA_N_TRIALS', 200))


# =============================================================================
# OPTUNA OBJECTIVE FUNCTIONS
# =============================================================================
def _xgb_eval_metric(objective):
    return 'mlogloss' if objective == 'multi:softprob' else 'logloss'


def _xgb_cls_objective(trial, X, y, cv, objective, sw):
    evm = _xgb_eval_metric(objective)
    p = {
        'n_estimators': MAX_EST,
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_float('lr', 0.005, 0.12, log=True),
        'min_child_weight': trial.suggest_int('mcw', 1, 10),
        'subsample': trial.suggest_float('sub', 0.65, 1.0),
        'colsample_bytree': trial.suggest_float('csbt', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 3.0),
        'reg_alpha': trial.suggest_float('alpha', 1e-4, 2.0, log=True),
        'reg_lambda': trial.suggest_float('lam', 0.3, 5.0),
    }
    scores, iters = [], []
    for tr_idx, va_idx in cv.split(X):
        Xt, Xv = X.iloc[tr_idx], X.iloc[va_idx]
        yt, yv = y.iloc[tr_idx], y.iloc[va_idx]
        wt = sw[tr_idx] if sw is not None else None
        m = XGBClassifier(objective=objective, eval_metric=evm,
                          early_stopping_rounds=ES, **_xgb_base(), **p)
        fit_kw = {'eval_set': [(Xv, yv)], 'verbose': False}
        if wt is not None:
            fit_kw['sample_weight'] = wt
        m.fit(Xt, yt, **fit_kw)
        proba = m.predict_proba(Xv)
        scores.append(log_loss(yv, proba))
        iters.append(m.best_iteration)
    trial.set_user_attr('best_iter', int(np.median(iters)))
    return float(np.mean(scores))


def _lgbm_cls_objective(trial, X, y, cv, objective, sw):
    is_multi = objective == 'multi:softprob'
    p = {
        'n_estimators': MAX_EST,
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('lr', 0.005, 0.12, log=True),
        'min_child_samples': trial.suggest_int('mcs', 5, 50),
        'subsample': trial.suggest_float('sub', 0.65, 1.0),
        'colsample_bytree': trial.suggest_float('csbt', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('alpha', 1e-4, 2.0, log=True),
        'reg_lambda': trial.suggest_float('lam', 0.3, 5.0),
        'num_leaves': trial.suggest_int('leaves', 15, 63),
    }
    lgbm_obj = 'multiclass' if is_multi else 'binary'
    scores, iters = [], []
    for tr_idx, va_idx in cv.split(X):
        Xt, Xv = X.iloc[tr_idx], X.iloc[va_idx]
        yt, yv = y.iloc[tr_idx], y.iloc[va_idx]
        wt = sw[tr_idx] if sw is not None else None
        m = LGBMClassifier(objective=lgbm_obj, n_jobs=-1, random_state=42,
                           verbose=-1, **p)
        fit_kw = {'eval_set': [(Xv, yv)],
                  'callbacks': [_lgbm_early_stopping(ES)]}
        if wt is not None:
            fit_kw['sample_weight'] = wt
        m.fit(Xt, yt, **fit_kw)
        proba = m.predict_proba(Xv)
        scores.append(log_loss(yv, proba))
        iters.append(m.best_iteration_ if hasattr(m, 'best_iteration_') else MAX_EST)
    trial.set_user_attr('best_iter', int(np.median(iters)))
    return float(np.mean(scores))


def _xgb_reg_objective(trial, X, y, cv, sw):
    p = {
        'n_estimators': MAX_EST,
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'learning_rate': trial.suggest_float('lr', 0.005, 0.1, log=True),
        'min_child_weight': trial.suggest_int('mcw', 1, 10),
        'subsample': trial.suggest_float('sub', 0.65, 1.0),
        'colsample_bytree': trial.suggest_float('csbt', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 3.0),
        'reg_alpha': trial.suggest_float('alpha', 1e-4, 2.0, log=True),
        'reg_lambda': trial.suggest_float('lam', 0.3, 5.0),
    }
    scores, iters = [], []
    for tr_idx, va_idx in cv.split(X):
        Xt, Xv = X.iloc[tr_idx], X.iloc[va_idx]
        yt, yv = y.iloc[tr_idx], y.iloc[va_idx]
        wt = sw[tr_idx] if sw is not None else None
        m = XGBRegressor(objective='reg:squarederror', eval_metric='mae',
                         early_stopping_rounds=ES, **_xgb_base(), **p)
        fit_kw = {'eval_set': [(Xv, yv)], 'verbose': False}
        if wt is not None:
            fit_kw['sample_weight'] = wt
        m.fit(Xt, yt, **fit_kw)
        scores.append(mean_absolute_error(yv, m.predict(Xv)))
        iters.append(m.best_iteration)
    trial.set_user_attr('best_iter', int(np.median(iters)))
    return float(np.mean(scores))


def _lgbm_reg_objective(trial, X, y, cv, sw):
    p = {
        'n_estimators': MAX_EST,
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_float('lr', 0.005, 0.1, log=True),
        'min_child_samples': trial.suggest_int('mcs', 5, 50),
        'subsample': trial.suggest_float('sub', 0.65, 1.0),
        'colsample_bytree': trial.suggest_float('csbt', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('alpha', 1e-4, 2.0, log=True),
        'reg_lambda': trial.suggest_float('lam', 0.3, 5.0),
        'num_leaves': trial.suggest_int('leaves', 15, 63),
    }
    scores, iters = [], []
    for tr_idx, va_idx in cv.split(X):
        Xt, Xv = X.iloc[tr_idx], X.iloc[va_idx]
        yt, yv = y.iloc[tr_idx], y.iloc[va_idx]
        wt = sw[tr_idx] if sw is not None else None
        m = LGBMRegressor(objective='regression', n_jobs=-1, random_state=42,
                          verbose=-1, **p)
        fit_kw = {'eval_set': [(Xv, yv)],
                  'callbacks': [_lgbm_early_stopping(ES)]}
        if wt is not None:
            fit_kw['sample_weight'] = wt
        m.fit(Xt, yt, **fit_kw)
        scores.append(mean_absolute_error(yv, m.predict(Xv)))
        iters.append(m.best_iteration_ if hasattr(m, 'best_iteration_') else MAX_EST)
    trial.set_user_attr('best_iter', int(np.median(iters)))
    return float(np.mean(scores))


def _lgbm_early_stopping(rounds):
    """Return a LightGBM early_stopping callback."""
    from lightgbm import early_stopping
    return early_stopping(stopping_rounds=rounds, verbose=False)


# =============================================================================
# TRAIN + CALIBRATE
# =============================================================================
def _train_cls(X_train, y_train, objective, sw, n_trials=None):
    """Optuna-tune XGB + LGBM classifiers, calibrate each, return both."""
    if n_trials is None:
        n_trials = N_TRIALS
    n = len(X_train)
    n_sp = _ts_splits(n)
    cv = TimeSeriesSplit(n_splits=n_sp)

    cs = _cal_split(n)
    if cs is not None:
        fe, _ = cs
        Xf, yf = X_train.iloc[:fe], y_train.iloc[:fe]
        Xc, yc = X_train.iloc[fe:], y_train.iloc[fe:]
        wf = sw[:fe] if sw is not None else None
    else:
        Xf, yf = X_train, y_train
        Xc, yc = None, None
        wf = sw

    cv_fit = TimeSeriesSplit(n_splits=_ts_splits(len(Xf)))

    models = {}
    for engine in config.MODEL_TYPES:
        if engine == 'xgb':
            obj_fn = lambda trial: _xgb_cls_objective(trial, Xf, yf, cv_fit, objective, wf)
        else:
            obj_fn = lambda trial: _lgbm_cls_objective(trial, Xf, yf, cv_fit, objective, wf)

        study = optuna.create_study(direction='minimize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(obj_fn, n_trials=n_trials, timeout=getattr(config, 'OPTUNA_TIMEOUT', None))

        bp = study.best_trial.params
        bi = study.best_trial.user_attrs.get('best_iter', MAX_EST)
        bi = max(bi, 50)

        evm = _xgb_eval_metric(objective)
        if engine == 'xgb':
            best = XGBClassifier(
                objective=objective, eval_metric=evm,
                n_estimators=bi, max_depth=bp['max_depth'],
                learning_rate=bp['lr'], min_child_weight=bp['mcw'],
                subsample=bp['sub'], colsample_bytree=bp['csbt'],
                gamma=bp['gamma'], reg_alpha=bp['alpha'],
                reg_lambda=bp['lam'], **_xgb_base(),
            )
        else:
            lgbm_obj = 'multiclass' if objective == 'multi:softprob' else 'binary'
            best = LGBMClassifier(
                objective=lgbm_obj, n_jobs=-1, random_state=42, verbose=-1,
                n_estimators=bi, max_depth=bp['max_depth'],
                learning_rate=bp['lr'], min_child_samples=bp['mcs'],
                subsample=bp['sub'], colsample_bytree=bp['csbt'],
                reg_alpha=bp['alpha'], reg_lambda=bp['lam'],
                num_leaves=bp['leaves'],
            )

        fit_kw = {}
        if wf is not None:
            fit_kw['sample_weight'] = wf
        best.fit(Xf, yf, **fit_kw)

        cal_model = _calibrate(best, Xc, yc)
        models[engine] = {
            'model': cal_model,
            'cv_score': study.best_value,
            'n_iters': bi,
            'calibrated': cal_model is not best,
        }

    return models


def _train_reg(X_train, y_train, sw, n_trials=None):
    """Optuna-tune XGB + LGBM regressors, return both."""
    if n_trials is None:
        n_trials = N_TRIALS
    cv = TimeSeriesSplit(n_splits=_ts_splits(len(X_train)))

    models = {}
    for engine in config.MODEL_TYPES:
        if engine == 'xgb':
            obj_fn = lambda trial: _xgb_reg_objective(trial, X_train, y_train, cv, sw)
        else:
            obj_fn = lambda trial: _lgbm_reg_objective(trial, X_train, y_train, cv, sw)

        study = optuna.create_study(direction='minimize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(obj_fn, n_trials=n_trials, timeout=getattr(config, 'OPTUNA_TIMEOUT', None))

        bp = study.best_trial.params
        bi = study.best_trial.user_attrs.get('best_iter', MAX_EST)
        bi = max(bi, 50)

        if engine == 'xgb':
            best = XGBRegressor(
                objective='reg:squarederror', n_estimators=bi,
                max_depth=bp['max_depth'], learning_rate=bp['lr'],
                min_child_weight=bp['mcw'], subsample=bp['sub'],
                colsample_bytree=bp['csbt'], gamma=bp['gamma'],
                reg_alpha=bp['alpha'], reg_lambda=bp['lam'],
                **_xgb_base(),
            )
        else:
            best = LGBMRegressor(
                objective='regression', n_jobs=-1, random_state=42, verbose=-1,
                n_estimators=bi, max_depth=bp['max_depth'],
                learning_rate=bp['lr'], min_child_samples=bp['mcs'],
                subsample=bp['sub'], colsample_bytree=bp['csbt'],
                reg_alpha=bp['alpha'], reg_lambda=bp['lam'],
                num_leaves=bp['leaves'],
            )

        fit_kw = {}
        if sw is not None:
            fit_kw['sample_weight'] = sw
        best.fit(X_train, y_train, **fit_kw)
        models[engine] = {'model': best, 'cv_score': study.best_value, 'n_iters': bi}

    return models


def _calibrate(model, X_cal, y_cal):
    """Wrap model with temporal holdout calibration if possible."""
    if X_cal is None or y_cal is None or len(y_cal) < 50:
        return model
    class_fit = set(pd.Series(y_cal).dropna().unique().tolist())
    if len(class_fit) < 2:
        return model
    method = _pick_cal_method(y_cal, str(getattr(config, 'CALIBRATION_METHOD', 'auto')))
    try:
        frozen = FrozenEstimator(model)
        cal = CalibratedClassifierCV(estimator=frozen, method=method)
        cal.fit(X_cal, y_cal)
        return cal
    except Exception:
        return model


# =============================================================================
# PER-MARKET TRAINING
# =============================================================================
MARKET_DEFS = {
    '1x2':   ('result_label',  'multi:softprob'),
    'ou25':  ('over_2_5',      'binary:logistic'),
    'ou15':  ('over_1_5',      'binary:logistic'),
    'ou35':  ('over_3_5',      'binary:logistic'),
    'btts':  ('btts',          'binary:logistic'),
    'ht1x2': ('ht_result',     'multi:softprob'),
    'htou05':('ht_over_0_5',   'binary:logistic'),
}


def _train_cls_market(market, target, objective, league, lt, feats, verbose,
                      done, decay, model_set, n_trials):
    for eng in config.MODEL_TYPES:
        ck_key = _ck(league, market, eng)
        if done is not None and ck_key in done:
            if verbose:
                print(f"    {market.upper()}/{eng}: skip (done)")
            continue

        sub = lt.dropna(subset=feats + [target])
        if len(sub) < 50:
            if verbose:
                print(f"    {market.upper()}/{eng}: skip ({len(sub)} rows)")
            if done is not None:
                done.add(ck_key); _save_checkpoint(done, model_set)
            continue

        X, y = sub[feats], sub[target]
        sw = _time_decay_weights(sub['Date'], decay)

        if verbose:
            print(f"    {market.upper()}/{eng}: {len(X)} rows, {n_trials} trials ...", end='', flush=True)

        t0 = time.time()
        n_sp = _ts_splits(len(X))
        cv = TimeSeriesSplit(n_splits=n_sp)
        cs = _cal_split(len(X))

        if cs:
            fe, _ = cs
            Xf, yf, Xc, yc = X.iloc[:fe], y.iloc[:fe], X.iloc[fe:], y.iloc[fe:]
            wf = sw[:fe] if sw is not None else None
        else:
            Xf, yf, Xc, yc = X, y, None, None
            wf = sw

        cv_fit = TimeSeriesSplit(n_splits=_ts_splits(len(Xf)))

        if eng == 'xgb':
            obj_fn = lambda trial: _xgb_cls_objective(trial, Xf, yf, cv_fit, objective, wf)
        else:
            obj_fn = lambda trial: _lgbm_cls_objective(trial, Xf, yf, cv_fit, objective, wf)

        study = optuna.create_study(direction='minimize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(obj_fn, n_trials=n_trials,
                       timeout=getattr(config, 'OPTUNA_TIMEOUT', None))

        bp = study.best_trial.params
        bi = max(study.best_trial.user_attrs.get('best_iter', MAX_EST), 50)

        evm = _xgb_eval_metric(objective)
        if eng == 'xgb':
            model = XGBClassifier(
                objective=objective, eval_metric=evm,
                n_estimators=bi, max_depth=bp['max_depth'],
                learning_rate=bp['lr'], min_child_weight=bp['mcw'],
                subsample=bp['sub'], colsample_bytree=bp['csbt'],
                gamma=bp['gamma'], reg_alpha=bp['alpha'],
                reg_lambda=bp['lam'], **_xgb_base(),
            )
        else:
            lgbm_obj = 'multiclass' if objective == 'multi:softprob' else 'binary'
            model = LGBMClassifier(
                objective=lgbm_obj, n_jobs=-1, random_state=42, verbose=-1,
                n_estimators=bi, max_depth=bp['max_depth'],
                learning_rate=bp['lr'], min_child_samples=bp['mcs'],
                subsample=bp['sub'], colsample_bytree=bp['csbt'],
                reg_alpha=bp['alpha'], reg_lambda=bp['lam'],
                num_leaves=bp['leaves'],
            )

        fit_kw = {}
        if wf is not None:
            fit_kw['sample_weight'] = wf
        model.fit(Xf, yf, **fit_kw)

        final = _calibrate(model, Xc, yc)
        utils.save_model(final, market, league, model_set=model_set, engine=eng)

        elapsed = time.time() - t0
        cal_note = 'cal' if final is not model else 'no-cal'
        if verbose:
            print(f"  saved [{elapsed:.0f}s, ll={study.best_value:.3f}, "
                  f"iters={bi}, {cal_note}]")

        if done is not None:
            done.add(ck_key); _save_checkpoint(done, model_set)


def _train_xg_market(league, lt, feats, verbose, done, decay, model_set, n_trials):
    sub = lt.dropna(subset=feats + ['FTHG', 'FTAG'])
    if len(sub) < 50:
        if verbose:
            print(f"    xG: skip ({len(sub)} rows)")
        for eng in config.MODEL_TYPES:
            for s in ('Home', 'Away'):
                ck_key = _ck(league, 'xg', eng, s)
                if done is not None:
                    done.add(ck_key); _save_checkpoint(done, model_set)
        return

    X = sub[feats]
    sw = _time_decay_weights(sub['Date'], decay)
    n_sp = _ts_splits(len(X))
    cv = TimeSeriesSplit(n_splits=n_sp)

    for col, mtype, label in [('FTHG', 'xGH', 'Home'), ('FTAG', 'xGA', 'Away')]:
        y = sub[col]
        for eng in config.MODEL_TYPES:
            ck_key = _ck(league, 'xg', eng, label)
            if done is not None and ck_key in done:
                if verbose:
                    print(f"    xG-{label}/{eng}: skip (done)")
                continue

            if verbose:
                print(f"    xG-{label}/{eng}: {len(X)} rows ...", end='', flush=True)

            t0 = time.time()
            if eng == 'xgb':
                obj_fn = lambda trial: _xgb_reg_objective(trial, X, y, cv, sw)
            else:
                obj_fn = lambda trial: _lgbm_reg_objective(trial, X, y, cv, sw)

            study = optuna.create_study(direction='minimize',
                                        sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(obj_fn, n_trials=n_trials,
                           timeout=getattr(config, 'OPTUNA_TIMEOUT', None))

            bp = study.best_trial.params
            bi = max(study.best_trial.user_attrs.get('best_iter', MAX_EST), 50)

            if eng == 'xgb':
                model = XGBRegressor(
                    objective='reg:squarederror', n_estimators=bi,
                    max_depth=bp['max_depth'], learning_rate=bp['lr'],
                    min_child_weight=bp['mcw'], subsample=bp['sub'],
                    colsample_bytree=bp['csbt'], gamma=bp['gamma'],
                    reg_alpha=bp['alpha'], reg_lambda=bp['lam'],
                    **_xgb_base(),
                )
            else:
                model = LGBMRegressor(
                    objective='regression', n_jobs=-1, random_state=42, verbose=-1,
                    n_estimators=bi, max_depth=bp['max_depth'],
                    learning_rate=bp['lr'], min_child_samples=bp['mcs'],
                    subsample=bp['sub'], colsample_bytree=bp['csbt'],
                    reg_alpha=bp['alpha'], reg_lambda=bp['lam'],
                    num_leaves=bp['leaves'],
                )

            fit_kw = {}
            if sw is not None:
                fit_kw['sample_weight'] = sw
            model.fit(X, y, **fit_kw)
            utils.save_model(model, mtype, league, model_set=model_set, engine=eng)

            elapsed = time.time() - t0
            if verbose:
                print(f"  saved [{elapsed:.0f}s, mae={study.best_value:.3f}, iters={bi}]")

            if done is not None:
                done.add(ck_key); _save_checkpoint(done, model_set)


# =============================================================================
# MAIN LOOP
# =============================================================================
def train_all_models(all_data, leagues=None, models_to_train=None,
                     verbose=True, fresh=False,
                     recent_days=None, time_decay_half_life_days=None,
                     model_set='current', n_trials=None):
    if models_to_train is None:
        models_to_train = config.ALL_MARKETS
    if n_trials is None:
        n_trials = N_TRIALS

    done = set() if fresh else _load_checkpoint(model_set=model_set)
    if done and verbose:
        print(f"\n  Resuming: {len(done)} models done (use --fresh to retrain)")

    if recent_days is None:
        recent_days = getattr(config, 'TRAIN_RECENT_DAYS', None)
    if time_decay_half_life_days is None:
        time_decay_half_life_days = getattr(config, 'TIME_DECAY_HALF_LIFE_DAYS', None)

    cutoff = config.TRAIN_CUTOFF
    if cutoff is not None:
        cutoff = pd.Timestamp(cutoff)
        train_data = all_data[(all_data['Date'] < cutoff) &
                              (all_data['Date'].dt.year >= config.TRAIN_START_YEAR)]
    else:
        train_data = all_data[all_data['Date'].dt.year >= config.TRAIN_START_YEAR]

    if recent_days is not None and int(recent_days) > 0:
        end_date = cutoff if cutoff is not None else train_data['Date'].max()
        train_data = train_data[train_data['Date'] >= end_date - pd.Timedelta(days=int(recent_days))]
    else:
        recent_days = None

    if verbose:
        d0, d1 = train_data['Date'].min().date(), train_data['Date'].max().date()
        print(f"\n  Train: {len(train_data)} matches ({d0} to {d1})")
        if time_decay_half_life_days and float(time_decay_half_life_days) > 0:
            print(f"  Decay half-life: {float(time_decay_half_life_days):g} days")
        print(f"  Optuna trials/model: {n_trials}")
        print(f"  Engines: {config.MODEL_TYPES}")
        print(f"  Early stopping: {ES} rounds, max {MAX_EST} iters")

    if leagues is None:
        leagues = sorted(l for l in all_data['league'].unique()
                         if l in config.LEAGUE_REGISTRY)

    total_start = time.time()
    for league in leagues:
        info = config.LEAGUE_REGISTRY.get(league, {})
        disp = info.get('display_name', league)
        is_rich = info.get('type') == 'rich'
        lt = train_data[train_data['league'] == league]

        if len(lt) < config.MIN_MATCHES_PER_LEAGUE:
            if verbose:
                print(f"\n  {disp}: skip ({len(lt)} rows)")
            continue

        if verbose:
            print(f"\n{'='*60}\n  {disp}  ({len(lt)} rows)\n{'='*60}")

        feats = config.get_features_for_league(league)

        for market in models_to_train:
            if market in config.RICH_ONLY_MARKETS and not is_rich:
                continue
            if market == 'xg':
                _train_xg_market(league, lt, feats, verbose, done,
                                 time_decay_half_life_days, model_set, n_trials)
            elif market in MARKET_DEFS:
                target, obj = MARKET_DEFS[market]
                _train_cls_market(market, target, obj, league, lt, feats,
                                  verbose, done, time_decay_half_life_days,
                                  model_set, n_trials)

    elapsed = time.time() - total_start
    if verbose:
        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)
        print(f"\n  Total training time: {h}h {m}m {s}s")


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================
def validate_models(all_data, validation_days=None, leagues=None,
                    models_to_validate=None, verbose=True,
                    time_decay_half_life_days=None, n_trials=None):
    if validation_days is None:
        validation_days = int(getattr(config, 'VALIDATION_DAYS', 60))
    if models_to_validate is None:
        models_to_validate = config.ALL_MARKETS
    if time_decay_half_life_days is None:
        time_decay_half_life_days = getattr(config, 'TIME_DECAY_HALF_LIFE_DAYS', None)
    if n_trials is None:
        n_trials = max(N_TRIALS // 2, 30)

    base = all_data[all_data['Date'].dt.year >= config.TRAIN_START_YEAR]
    max_date = base['Date'].max()
    cutoff = max_date - pd.Timedelta(days=validation_days)
    train_d = base[base['Date'] < cutoff]
    test_d = base[base['Date'] >= cutoff]

    if verbose:
        print(f"\n{'='*60}\n  WALK-FORWARD VALIDATION\n{'='*60}")
        print(f"  Train: {len(train_d)} ({cutoff.date()} cutoff)")
        print(f"  Test:  {len(test_d)} ({cutoff.date()} to {max_date.date()})")

    if leagues is None:
        leagues = sorted(l for l in base['league'].unique()
                         if l in config.LEAGUE_REGISTRY)

    results = []
    total_start = time.time()

    for league in leagues:
        info = config.LEAGUE_REGISTRY.get(league, {})
        disp = info.get('display_name', league)
        is_rich = info.get('type') == 'rich'
        lt = train_d[train_d['league'] == league]
        le = test_d[test_d['league'] == league]

        if len(lt) < config.MIN_MATCHES_PER_LEAGUE or len(le) < 5:
            continue

        feats = config.get_features_for_league(league)
        if verbose:
            print(f"\n  {disp}  ({len(lt)} train / {len(le)} test)")

        for market in models_to_validate:
            if market in config.RICH_ONLY_MARKETS and not is_rich:
                continue

            if market == 'xg':
                for col, label in [('FTHG', 'xG-Home'), ('FTAG', 'xG-Away')]:
                    tr = lt.dropna(subset=feats + [col])
                    te = le.dropna(subset=feats + [col])
                    if len(tr) < 50 or len(te) < 5:
                        continue
                    sw = _time_decay_weights(tr['Date'], time_decay_half_life_days)
                    reg_models = _train_reg(tr[feats], tr[col], sw, n_trials=n_trials)
                    preds = np.mean([m['model'].predict(te[feats])
                                     for m in reg_models.values()], axis=0)
                    mae = mean_absolute_error(te[col], preds)
                    results.append({'league': disp, 'market': label,
                                    'accuracy': None, 'log_loss': None,
                                    'mae': mae, 'n_test': len(te)})
                    if verbose:
                        print(f"    {label}: MAE={mae:.3f} (n={len(te)})")

            elif market in MARKET_DEFS:
                target, obj = MARKET_DEFS[market]
                tr = lt.dropna(subset=feats + [target])
                te = le.dropna(subset=feats + [target])
                if len(tr) < 50 or len(te) < 5:
                    continue
                sw = _time_decay_weights(tr['Date'], time_decay_half_life_days)
                cls_models = _train_cls(tr[feats], tr[target], obj, sw,
                                        n_trials=n_trials)
                probas = [m['model'].predict_proba(te[feats])
                          for m in cls_models.values()]
                avg_p = np.mean(probas, axis=0)
                pred = np.argmax(avg_p, axis=1)
                acc = accuracy_score(te[target], pred)
                try:
                    ll = log_loss(te[target], avg_p)
                except Exception:
                    ll = float('nan')
                results.append({'league': disp, 'market': market.upper(),
                                'accuracy': acc, 'log_loss': ll,
                                'mae': None, 'n_test': len(te)})
                if verbose:
                    print(f"    {market.upper()}: acc={acc:.1%} ll={ll:.3f} (n={len(te)})")

    elapsed = time.time() - total_start
    if verbose:
        _print_val_summary(results, elapsed)
    return results


def _print_val_summary(results, elapsed):
    if not results:
        print("\n  No results.")
        return
    df = pd.DataFrame(results)
    print(f"\n{'='*60}\n  VALIDATION SUMMARY\n{'='*60}")
    print(f"  {'Market':<12} {'Accuracy':>10} {'LogLoss':>10} {'MAE':>8} {'N':>8}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    for market in df['market'].unique():
        mdf = df[df['market'] == market]
        n_tot = int(mdf['n_test'].sum())
        if mdf['accuracy'].notna().any():
            w = mdf['n_test']
            wa = np.average(mdf['accuracy'].dropna(), weights=w[mdf['accuracy'].notna()])
            wl = np.average(mdf['log_loss'].dropna(), weights=w[mdf['log_loss'].notna()])
            print(f"  {market:<12} {wa:>9.1%} {wl:>10.3f} {'':>8} {n_tot:>8}")
        elif mdf['mae'].notna().any():
            wm = np.average(mdf['mae'].dropna(), weights=mdf['n_test'][mdf['mae'].notna()])
            print(f"  {market:<12} {'':>10} {'':>10} {wm:>7.3f} {n_tot:>8}")
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    print(f"\n  Validation time: {h}h {m}m {s}s\n{'='*60}")


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Train football prediction models')
    parser.add_argument('--models', nargs='+',
                        choices=config.ALL_MARKETS + ['all'], default=['all'])
    parser.add_argument('--leagues', nargs='+')
    parser.add_argument('--no-save-data', action='store_true')
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument('--recent-days', type=int, default=None)
    parser.add_argument('--decay-half-life-days', type=float, default=None)
    parser.add_argument('--no-time-decay', action='store_true')
    parser.add_argument('--model-set', type=str, default='current')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--validate-days', type=int, default=None)
    parser.add_argument('--n-trials', type=int, default=None,
                        help=f'Optuna trials per model (default {N_TRIALS})')
    args = parser.parse_args()

    print("=" * 60)
    print("FOOTBALL PREDICTOR V2 - MODEL TRAINING")
    print("=" * 60)

    print("\nLoading and processing data...")
    all_data = data_loader.load_and_process_all_leagues(verbose=True)
    n = len(all_data)
    d0, d1 = all_data['Date'].min().date(), all_data['Date'].max().date()
    nl = all_data['league'].nunique()
    print(f"\n  Total: {n} matches | {nl} leagues | {d0} to {d1}")

    if not args.no_save_data:
        all_data.to_csv(config.PROCESSED_DATA_FILE, index=False)
        print(f"  Saved -> {config.PROCESSED_DATA_FILE}")

    models = args.models
    if 'all' in models:
        models = config.ALL_MARKETS

    decay = 0.0 if args.no_time_decay else args.decay_half_life_days

    if args.validate or args.validate_only:
        validate_models(all_data, validation_days=args.validate_days,
                        leagues=args.leagues, models_to_validate=models,
                        time_decay_half_life_days=decay,
                        n_trials=args.n_trials)
        if args.validate_only:
            print("\n  Validation complete (no models saved).")
            return

    train_all_models(all_data, leagues=args.leagues,
                     models_to_train=models, verbose=True,
                     fresh=args.fresh, recent_days=args.recent_days,
                     time_decay_half_life_days=decay,
                     model_set=args.model_set, n_trials=args.n_trials)

    clear_checkpoint(model_set=args.model_set)
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == '__main__':
    main()
