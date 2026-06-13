"""
Probability Blending Layer (1X2)
=================================
Log-pools three probability sources per match:
  - GBM ensemble (xgb + lgbm averaged, mirroring predict.py)
  - bookmaker implied probabilities (normalised 1/odds)
  - Dixon-Coles baseline (refit monthly, leak-free)

blended  p  ∝  p_model^w1 · p_book^w2 · p_dc^w3      (weights >= 0)

Unconstrained weight sums act as a temperature, so the fit also handles
over/under-confidence. Weights are fitted on the first part of the
out-of-sample window and evaluated on the rest — never on training data.

Usage:
    python scripts/blend.py evaluate [--cutoff 2026-02-01] [--split 0.5]
    python scripts/blend.py fit      [--cutoff 2026-02-01]   # fit on full OOS
                                                             # window and save
"""

import argparse
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, log_loss

from scripts import config
from scripts import data_loader
from scripts import utils
from scripts.dixon_coles import DixonColesModel

warnings.filterwarnings('ignore')

BLEND_WEIGHTS_FILE = os.path.join(config.MODELS_DIR, 'blend_weights.json')

# Leagues whose models were trained on the full dataset (no OOS window yet);
# excluded from weight fitting/evaluation until their next cutoff retrain.
IN_SAMPLE_LEAGUES = {'usa', 'mexico', 'irish'}

EPS = 1e-9


# =============================================================================
# COMPONENT PROBABILITIES
# =============================================================================
def ensemble_proba(league: str, X: pd.DataFrame) -> np.ndarray | None:
    """Average xgb + lgbm predict_proba, as predict.py does."""
    probas = []
    for eng in config.MODEL_TYPES:
        m = utils.load_model('1x2', league, engine=eng)
        if m is not None:
            try:
                probas.append(m.predict_proba(X))
            except Exception:
                pass
    if not probas:
        return None
    return np.mean(probas, axis=0)


def ensemble_proba_market(market: str, league: str,
                          X: pd.DataFrame) -> np.ndarray | None:
    """Engine-averaged predict_proba for any binary market."""
    probas = []
    for eng in config.MODEL_TYPES:
        m = utils.load_model(market, league, engine=eng)
        if m is not None:
            try:
                probas.append(m.predict_proba(X))
            except Exception:
                pass
    if not probas:
        return None
    return np.mean(probas, axis=0)


def implied_proba(df: pd.DataFrame) -> np.ndarray:
    """Normalised 1/odds in class order (A, D, H). NaN rows stay NaN."""
    inv = np.column_stack([
        1.0 / df['OddsA'].to_numpy(float),
        1.0 / df['OddsD'].to_numpy(float),
        1.0 / df['OddsH'].to_numpy(float),
    ])
    return inv / inv.sum(axis=1, keepdims=True)


def dc_proba(league_df: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    """Dixon-Coles probabilities for `target` matches (class order A, D, H).

    Refits at each month boundary using only matches strictly before it.
    Rows with unseen teams or failed fits come back as NaN.
    """
    out = np.full((len(target), 3), np.nan)
    target = target.sort_values('Date')
    months = target['Date'].dt.to_period('M')
    pos = {idx: k for k, idx in enumerate(target.index)}

    for month in months.unique():
        block = target[months == month]
        model = DixonColesModel().fit(league_df, as_of=month.to_timestamp())
        if model is None:
            continue
        for idx, m in block.iterrows():
            h, a = m['HomeTeam'], m['AwayTeam']
            if not (model.knows(h) and model.knows(a)):
                continue
            p = model.market_probs(h, a)
            out[pos[idx]] = [p['p_away'], p['p_draw'], p['p_home']]
    return out


# =============================================================================
# LOG-POOL BLEND
# =============================================================================
def log_pool(components: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """p ∝ Π p_i^w_i over available (non-NaN) components per row."""
    n, k = components[0].shape
    log_p = np.zeros((n, k))
    for comp, w in zip(components, weights):
        c = np.log(np.clip(np.nan_to_num(comp, nan=1.0 / 3), EPS, 1.0))
        valid = ~np.isnan(comp[:, 0])
        log_p[valid] += w * c[valid]
    log_p -= log_p.max(axis=1, keepdims=True)
    p = np.exp(log_p)
    return p / p.sum(axis=1, keepdims=True)


def fit_weights(components: list[np.ndarray], y: np.ndarray,
                x0=None) -> np.ndarray:
    k = len(components)
    if x0 is None:
        x0 = np.full(k, 1.0 / k)

    idx = np.arange(len(y))

    def loss(w):
        p = log_pool(components, w)
        return -np.log(np.clip(p[idx, y], EPS, 1.0)).mean()

    res = minimize(loss, x0, method='L-BFGS-B', bounds=[(0.0, 3.0)] * k)
    return res.x


SHRINK_K = 150   # pseudo-matches pulling per-league weights toward global


def fit_league_weights(df: pd.DataFrame, w_global: np.ndarray) -> dict:
    """Per-league weights, shrunk toward the global fit.

    w_league = (n·w_fit + K·w_global) / (n + K) — leagues with little data
    stay near the global weights; only persistent league-level edges move.
    """
    out = {}
    for league, g in df.groupby('league'):
        w_fit = fit_weights(_comps(g), g['y'].to_numpy(), x0=w_global.copy())
        n = len(g)
        out[league] = (n * w_fit + SHRINK_K * w_global) / (n + SHRINK_K)
    return out


def _league_blend(df: pd.DataFrame, weights_by_league: dict,
                  w_global: np.ndarray) -> np.ndarray:
    p = np.empty((len(df), 3))
    comps = _comps(df)
    leagues = df['league'].to_numpy()
    for league in np.unique(leagues):
        mask = leagues == league
        w = weights_by_league.get(league, w_global)
        sub = [c[mask] for c in comps]
        p[mask] = log_pool(sub, w)
    return p


# =============================================================================
# EVALUATION HARNESS
# =============================================================================
def build_components(all_data: pd.DataFrame, cutoff: str):
    """Collect per-match component probabilities for the OOS window."""
    cutoff = pd.Timestamp(cutoff)
    rows = []
    leagues = [l for l in utils.get_available_leagues()
               if l not in IN_SAMPLE_LEAGUES]

    for league in leagues:
        feats = config.get_features_for_league(league)
        ldf = all_data[all_data['league'] == league]
        test = ldf[(ldf['Date'] >= cutoff)].dropna(
            subset=feats + ['result_label', 'OddsH', 'OddsD', 'OddsA'])
        if len(test) < 10:
            continue

        p_model = ensemble_proba(league, test[feats])
        if p_model is None:
            continue
        p_book = implied_proba(test)
        p_dc = dc_proba(ldf, test)

        frame = pd.DataFrame({
            'league': league,
            'Date': test['Date'].to_numpy(),
            'y': test['result_label'].to_numpy(int),
            'mA': p_model[:, 0], 'mD': p_model[:, 1], 'mH': p_model[:, 2],
            'bA': p_book[:, 0], 'bD': p_book[:, 1], 'bH': p_book[:, 2],
            'dA': p_dc[:, 0], 'dD': p_dc[:, 1], 'dH': p_dc[:, 2],
        })

        # OU 2.5 components (model + book), where both exist
        frame['y_ou'] = np.nan
        frame['ouM'] = np.nan
        frame['ouB'] = np.nan
        p_ou = ensemble_proba_market('ou25', league, test[feats])
        if p_ou is not None and 'over_2_5' in test.columns:
            inv_o = 1.0 / test['OddsOver25'].to_numpy(float)
            inv_u = 1.0 / test['OddsUnder25'].to_numpy(float)
            book_over = inv_o / (inv_o + inv_u)
            frame['y_ou'] = test['over_2_5'].to_numpy(float)
            frame['ouM'] = p_ou[:, 1]
            frame['ouB'] = book_over

        rows.append(frame)

    return pd.concat(rows, ignore_index=True).sort_values('Date')


def _metrics(p, y):
    ll = log_loss(y, p, labels=[0, 1, 2])
    acc = accuracy_score(y, p.argmax(axis=1))
    return acc, ll


def _comps(df):
    return [df[['mA', 'mD', 'mH']].to_numpy(),
            df[['bA', 'bD', 'bH']].to_numpy(),
            df[['dA', 'dD', 'dH']].to_numpy()]


def cmd_evaluate(args):
    print("Loading data + building component probabilities "
          "(DC refits take a minute)...")
    all_data = data_loader.load_processed_data()
    data = build_components(all_data, args.cutoff)

    split_at = data['Date'].quantile(args.split)
    tune = data[data['Date'] < split_at]
    test = data[data['Date'] >= split_at]
    print(f"\nOOS window from {args.cutoff} | tune n={len(tune)} "
          f"(to {split_at.date()}) | eval n={len(test)}")

    yt = tune['y'].to_numpy()
    ye = test['y'].to_numpy()
    ct, ce = _comps(tune), _comps(test)

    strategies = {
        'model only':        np.array([1.0, 0.0, 0.0]),
        'book only':         np.array([0.0, 1.0, 0.0]),
        'DC only':           np.array([0.0, 0.0, 1.0]),
    }
    w_mb = fit_weights(ct[:2], yt)
    w_md = fit_weights([ct[0], ct[2]], yt)
    w_all = fit_weights(ct, yt)
    strategies['model+DC (fit)'] = np.array([w_md[0], 0.0, w_md[1]])
    strategies['model+book (fit)'] = np.array([w_mb[0], w_mb[1], 0.0])
    strategies['model+book+DC (fit)'] = w_all

    print(f"\n  {'Strategy':<22} {'Weights (m/b/dc)':<22} "
          f"{'Acc':>7} {'LogLoss':>9}")
    print('  ' + '-' * 62)
    for name, w in strategies.items():
        p = log_pool(ce, w)
        acc, ll = _metrics(p, ye)
        wtxt = '/'.join(f"{x:.2f}" for x in w)
        print(f"  {name:<22} {wtxt:<22} {acc:>7.1%} {ll:>9.4f}")

    lw = fit_league_weights(tune, w_all)
    p = _league_blend(test, lw, w_all)
    acc, ll = _metrics(p, ye)
    print(f"  {'per-league shrunk':<22} {'(varies)':<22} {acc:>7.1%} {ll:>9.4f}")

    # ----- OU 2.5 -----
    tune_ou = tune.dropna(subset=['y_ou', 'ouM', 'ouB'])
    test_ou = test.dropna(subset=['y_ou', 'ouM', 'ouB'])
    if len(tune_ou) > 100 and len(test_ou) > 100:
        def ou2(df):
            over_m = df['ouM'].to_numpy()
            over_b = df['ouB'].to_numpy()
            return ([np.column_stack([1 - over_m, over_m]),
                     np.column_stack([1 - over_b, over_b])],
                    df['y_ou'].to_numpy(int))
        ct_ou, yt_ou = ou2(tune_ou)
        ce_ou, ye_ou = ou2(test_ou)
        w_ou = fit_weights(ct_ou, yt_ou, x0=np.array([0.5, 0.5]))
        print(f"\n  OU 2.5 (eval n={len(test_ou)})")
        for name, w in [('model only', np.array([1.0, 0.0])),
                        ('book only', np.array([0.0, 1.0])),
                        ('model+book (fit)', w_ou)]:
            p = log_pool(ce_ou, w)
            ll = log_loss(ye_ou, p, labels=[0, 1])
            acc = accuracy_score(ye_ou, p.argmax(axis=1))
            wtxt = '/'.join(f"{x:.2f}" for x in w)
            print(f"  {name:<22} {wtxt:<22} {acc:>7.1%} {ll:>9.4f}")

    # per-league breakdown of the full blend vs book
    print(f"\n  {'League':<22} {'N':>5} {'Blend LL':>9} {'Book LL':>9} "
          f"{'Model LL':>9}")
    print('  ' + '-' * 58)
    for league, g in test.groupby('league'):
        if len(g) < 15:
            continue
        cg = _comps(g)
        yg = g['y'].to_numpy()
        ll_blend = _metrics(log_pool(cg, w_all), yg)[1]
        ll_book = _metrics(log_pool(cg, strategies['book only']), yg)[1]
        ll_model = _metrics(log_pool(cg, strategies['model only']), yg)[1]
        flag = ' <' if ll_blend < ll_book else ''
        print(f"  {league:<22} {len(g):>5} {ll_blend:>9.4f} "
              f"{ll_book:>9.4f} {ll_model:>9.4f}{flag}")


def cmd_fit(args):
    print("Loading data + building component probabilities...")
    all_data = data_loader.load_processed_data()
    data = build_components(all_data, args.cutoff)
    y = data['y'].to_numpy()
    comps = _comps(data)

    w_all = fit_weights(comps, y)
    w_no_book = fit_weights([comps[0], comps[2]], y)

    weights = {
        'with_odds': {'model': w_all[0], 'book': w_all[1], 'dc': w_all[2]},
        'no_odds': {'model': w_no_book[0], 'dc': w_no_book[1]},
        'fitted_on': f"{args.cutoff}..{str(data['Date'].max().date())}",
        'n_matches': int(len(data)),
    }

    data_ou = data.dropna(subset=['y_ou', 'ouM', 'ouB'])
    if len(data_ou) > 200:
        over_m = data_ou['ouM'].to_numpy()
        over_b = data_ou['ouB'].to_numpy()
        w_ou = fit_weights(
            [np.column_stack([1 - over_m, over_m]),
             np.column_stack([1 - over_b, over_b])],
            data_ou['y_ou'].to_numpy(int), x0=np.array([0.5, 0.5]))
        weights['ou25_with_odds'] = {'model': w_ou[0], 'book': w_ou[1],
                                     'n_matches': int(len(data_ou))}
    with open(BLEND_WEIGHTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(weights, f, indent=2)
    print(json.dumps(weights, indent=2))
    print(f"\nSaved -> {BLEND_WEIGHTS_FILE}")


def main():
    parser = argparse.ArgumentParser(description='1X2 probability blending')
    sub = parser.add_subparsers(dest='command', required=True)

    p_eval = sub.add_parser('evaluate', help='Tune/eval split comparison')
    p_eval.add_argument('--cutoff', default='2026-02-01',
                        help='Start of the model OOS window (train cutoff)')
    p_eval.add_argument('--split', type=float, default=0.5,
                        help='Fraction of OOS window used for weight tuning')

    p_fit = sub.add_parser('fit', help='Fit weights on full OOS window and save')
    p_fit.add_argument('--cutoff', default='2026-02-01')

    args = parser.parse_args()
    {'evaluate': cmd_evaluate, 'fit': cmd_fit}[args.command](args)


if __name__ == '__main__':
    main()
