import json
import os
import sys
from itertools import product

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts import config
from scripts.data_loader import load_league_data


def _expected_outcome(r1: np.ndarray | float, r2: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + 10.0 ** ((r2 - r1) / 400.0))


def calculate_elo_fast(
    df: pd.DataFrame,
    k: float,
    initial: float,
    season_regression: float,
    margin_factor: float,
) -> pd.DataFrame:
    df = df.sort_values('Date').reset_index(drop=True)
    n = len(df)
    if n == 0:
        return df

    teams = sorted(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))
    n_teams = len(teams)
    tix = {t: i for i, t in enumerate(teams)}

    hi = df['HomeTeam'].map(tix).to_numpy(np.int64, copy=False)
    ai = df['AwayTeam'].map(tix).to_numpy(np.int64, copy=False)
    fthg = df['FTHG'].to_numpy(np.int64, copy=False)
    ftag = df['FTAG'].to_numpy(np.int64, copy=False)
    dates = df['Date'].to_numpy(dtype='datetime64[ns]')

    att = np.full(n_teams, float(initial), dtype=np.float64)
    defe = np.full(n_teams, float(initial), dtype=np.float64)
    last_played = np.full(n_teams, np.datetime64('NaT'), dtype='datetime64[ns]')

    h_att = np.empty(n, dtype=np.float64)
    h_def = np.empty(n, dtype=np.float64)
    a_att = np.empty(n, dtype=np.float64)
    a_def = np.empty(n, dtype=np.float64)

    for i in range(n):
        h, a = hi[i], ai[i]
        dcur = dates[i]

        mean_att = float(att.mean())
        mean_def = float(defe.mean())

        if season_regression > 0:
            lp_h = last_played[h]
            if not np.isnat(lp_h):
                gap_h = (dcur - lp_h) / np.timedelta64(1, 'D')
                if gap_h > 60:
                    att[h] += season_regression * (mean_att - att[h])
                    defe[h] += season_regression * (mean_def - defe[h])
            lp_a = last_played[a]
            if not np.isnat(lp_a):
                gap_a = (dcur - lp_a) / np.timedelta64(1, 'D')
                if gap_a > 60:
                    att[a] += season_regression * (mean_att - att[a])
                    defe[a] += season_regression * (mean_def - defe[a])

        ha, hd = att[h], defe[h]
        aa, ad = att[a], defe[a]

        h_att[i], h_def[i] = ha, hd
        a_att[i], a_def[i] = aa, ad

        g_h, g_a = int(fthg[i]), int(ftag[i])
        total = g_h + g_a if (g_h + g_a) > 0 else 1
        margin = abs(g_h - g_a)
        ke = k * (1.0 + margin_factor * margin)

        e_home_att = _expected_outcome(ha, ad)
        e_away_att = _expected_outcome(aa, hd)
        s_home_att = g_h / total
        s_away_att = g_a / total

        att[h] = ha + ke * (s_home_att - e_home_att)
        att[a] = aa + ke * (s_away_att - e_away_att)

        e_home_def = _expected_outcome(hd, aa)
        e_away_def = _expected_outcome(ad, ha)
        s_home_def = 1.0 - (g_a / total)
        s_away_def = 1.0 - (g_h / total)

        defe[h] = hd + ke * (s_home_def - e_home_def)
        defe[a] = ad + ke * (s_away_def - e_away_def)

        last_played[h] = dcur
        last_played[a] = dcur

    out = df.copy()
    out['HomeAttackElo'] = h_att
    out['HomeDefenseElo'] = h_def
    out['AwayAttackElo'] = a_att
    out['AwayDefenseElo'] = a_def
    out['AttackElo_diff'] = h_att - a_att
    return out


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 10:
        return float('nan')
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float('nan')
    r = np.corrcoef(a, b)[0, 1]
    return float(r) if np.isfinite(r) else float('nan')


def backtest_elo_params(
    all_raw_data: dict[str, pd.DataFrame],
    verbose: bool = True,
) -> dict:
    combos = list(
        product(
            config.ELO_BACKTEST_K,
            config.ELO_BACKTEST_INITIAL,
            config.ELO_BACKTEST_REGRESSION,
            config.ELO_BACKTEST_MARGIN,
        )
    )
    n_combos = len(combos)
    best_score = -np.inf
    best_params: dict | None = None

    for j, (k, initial, reg, margin) in enumerate(combos, start=1):
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        gs: list[np.ndarray] = []

        for league, raw in all_raw_data.items():
            sub = raw[raw['Date'].dt.year >= config.ELO_WARMUP_YEAR].copy()
            if sub.empty:
                continue
            sub = calculate_elo_fast(sub.reset_index(drop=True), k, initial, reg, margin)
            lbl = sub['FTR'].map({'H': 2, 'D': 1, 'A': 0})
            mask = lbl.notna() & sub['AttackElo_diff'].notna()
            if not mask.any():
                continue
            tg = (sub['FTHG'] + sub['FTAG']).astype(float)
            xs.append(sub.loc[mask, 'AttackElo_diff'].to_numpy(dtype=np.float64))
            ys.append(lbl.loc[mask].to_numpy(dtype=np.float64))
            gs.append(tg.loc[mask].to_numpy(dtype=np.float64))

        if not xs:
            score = -np.inf
            r_res = float('nan')
            r_goals = float('nan')
        else:
            x = np.concatenate(xs)
            y = np.concatenate(ys)
            g = np.concatenate(gs)
            r_res = _pearson(x, y)
            r_goals = _pearson(x, g)
            if not np.isfinite(r_res):
                r_res = 0.0
            if not np.isfinite(r_goals):
                r_goals = 0.0
            score = abs(r_res) + abs(r_goals)

        if score > best_score:
            best_score = score
            best_params = {
                'k_factor': int(k) if float(k).is_integer() else float(k),
                'initial_rating': float(initial),
                'season_regression': float(reg),
                'margin_factor': float(margin),
                'score': float(score),
                'corr_result': float(r_res) if np.isfinite(r_res) else None,
                'corr_total_goals': float(r_goals) if np.isfinite(r_goals) else None,
            }

        if verbose:
            cr = r_res if np.isfinite(r_res) else float('nan')
            cg = r_goals if np.isfinite(r_goals) else float('nan')
            print(
                f"[{j}/{n_combos}] k={k} initial={initial} reg={reg} margin={margin} | "
                f"corr(AttackElo_diff,1x2)={cr:.4f} corr(AttackElo_diff,goals)={cg:.4f} | "
                f"combined={score:.4f}"
            )

    if best_params is None:
        best_params = {
            'k_factor': config.ELO_K_FACTOR,
            'initial_rating': float(config.ELO_INITIAL),
            'season_regression': float(config.ELO_SEASON_REGRESSION),
            'margin_factor': float(config.ELO_MARGIN_FACTOR),
        }
        if verbose:
            print('No valid backtest run; using config defaults.')
    else:
        save_payload = {k: best_params[k] for k in (
            'k_factor', 'initial_rating', 'season_regression', 'margin_factor',
        )}
        os.makedirs(os.path.dirname(config.ELO_PARAMS_FILE), exist_ok=True)
        with open(config.ELO_PARAMS_FILE, 'w', encoding='utf-8') as f:
            json.dump(save_payload, f, indent=2)
        if verbose:
            print()
            print('Best parameters:')
            for key in ('k_factor', 'initial_rating', 'season_regression', 'margin_factor'):
                print(f"  {key}: {save_payload[key]}")
            print(f"  combined score: {best_params['score']:.4f}")
            print(f"  corr vs 1x2 label: {best_params.get('corr_result')}")
            print(f"  corr vs total goals: {best_params.get('corr_total_goals')}")
            print(f"  saved to {config.ELO_PARAMS_FILE}")

    return best_params


def load_elo_params() -> dict:
    defaults = {
        'k_factor': config.ELO_K_FACTOR,
        'initial_rating': float(config.ELO_INITIAL),
        'season_regression': float(config.ELO_SEASON_REGRESSION),
        'margin_factor': float(config.ELO_MARGIN_FACTOR),
    }
    path = config.ELO_PARAMS_FILE
    if not os.path.isfile(path):
        return defaults.copy()
    try:
        with open(path, encoding='utf-8') as f:
            saved = json.load(f)
    except (json.JSONDecodeError, OSError):
        return defaults.copy()
    out = defaults.copy()
    for key in defaults:
        if key in saved:
            out[key] = type(defaults[key])(saved[key])
    return out


if __name__ == '__main__':
    raw_by_league: dict[str, pd.DataFrame] = {}
    for league_name in config.LEAGUE_REGISTRY:
        loaded = load_league_data(league_name, verbose=False)
        if not loaded.empty:
            raw_by_league[league_name] = loaded
    print(f"Loaded {len(raw_by_league)} leagues for Elo backtest.\n")
    backtest_elo_params(raw_by_league, verbose=True)
