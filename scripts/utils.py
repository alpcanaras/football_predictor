"""
Utility Functions
=================
Stats tables, model I/O, fixture feature construction, display helpers.
"""

import os
import glob
import pandas as pd
import numpy as np
import joblib
from . import config


# =============================================================================
# TEAM -> LEAGUE MAPPING
# =============================================================================
def get_team_to_league_map(df: pd.DataFrame) -> dict:
    df_sorted = df.sort_values('Date', ascending=False)
    team_league: dict[str, str] = {}
    for _, row in df_sorted.iterrows():
        if row['HomeTeam'] not in team_league:
            team_league[row['HomeTeam']] = row['league']
        if row['AwayTeam'] not in team_league:
            team_league[row['AwayTeam']] = row['league']
    return team_league


# =============================================================================
# LEAGUE-LEVEL HELPERS (precomputed once, shared across teams)
# =============================================================================
def _season_key(d) -> int:
    y, m = int(d.year), int(d.month)
    return y if m >= 8 else y - 1


def _build_league_tables(df: pd.DataFrame) -> dict:
    """Build current-season league tables for every league.

    Returns {league: {team: {'pts':, 'pld':, 'gf':, 'ga':, 'norm_pos':, 'ppg':}}}
    """
    tables: dict[str, dict] = {}
    for league in df['league'].unique():
        ldf = df[df['league'] == league].sort_values('Date')
        if ldf.empty:
            continue
        latest = ldf['Date'].max()
        season = _season_key(latest)
        sdf = ldf[ldf['Date'].apply(_season_key) == season]
        if sdf.empty:
            continue

        table: dict[str, dict] = {}
        for _, r in sdf.iterrows():
            ht, at = r['HomeTeam'], r['AwayTeam']
            for t in (ht, at):
                if t not in table:
                    table[t] = {'pts': 0, 'pld': 0, 'gf': 0, 'ga': 0}
            fthg, ftag = int(r['FTHG']), int(r['FTAG'])
            table[ht]['pld'] += 1
            table[at]['pld'] += 1
            table[ht]['gf'] += fthg
            table[ht]['ga'] += ftag
            table[at]['gf'] += ftag
            table[at]['ga'] += fthg
            res = r['FTR']
            if res == 'H':
                table[ht]['pts'] += 3
            elif res == 'A':
                table[at]['pts'] += 3
            else:
                table[ht]['pts'] += 1
                table[at]['pts'] += 1

        ranked = sorted(
            table.items(),
            key=lambda x: (-x[1]['pts'], -(x[1]['gf'] - x[1]['ga']), -x[1]['gf']),
        )
        n = len(ranked)
        for i, (team, info) in enumerate(ranked):
            if info['pld'] < 3:
                info['norm_pos'] = 0.5
            else:
                info['norm_pos'] = i / (n - 1) if n > 1 else 0.5
            info['ppg'] = info['pts'] / info['pld'] if info['pld'] > 0 else 0.0
        tables[league] = table
    return tables


def _league_home_rates(df: pd.DataFrame) -> dict:
    """Rolling home-win rate per league (last LEAGUE_HOME_WIN_WINDOW matches)."""
    hw = getattr(config, 'LEAGUE_HOME_WIN_WINDOW', 150)
    rates: dict[str, float] = {}
    for league in df['league'].unique():
        ldf = df[df['league'] == league].sort_values('Date').tail(hw)
        rates[league] = float((ldf['FTR'] == 'H').mean()) if len(ldf) > 0 else 0.0
    return rates


# =============================================================================
# TEAM STATS SNAPSHOT (used at prediction time)
# =============================================================================
def get_team_stats_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build a lookup table with each team's latest rolling stats."""
    all_teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())

    window = config.ROLLING_WINDOW
    sw = config.SHORT_WINDOW
    fw = config.FORM_WINDOW

    league_tables = _build_league_tables(df)
    home_rates = _league_home_rates(df)

    stats: list[dict] = []

    for team in all_teams:
        tm = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].sort_values('Date')
        if tm.empty:
            continue

        last = tm.iloc[-1]
        is_home_last = last['HomeTeam'] == team
        league = last['league']

        if 'FinalHomeAttackElo' in last.index:
            att_elo = last['FinalHomeAttackElo'] if is_home_last else last['FinalAwayAttackElo']
            def_elo = last['FinalHomeDefenseElo'] if is_home_last else last['FinalAwayDefenseElo']
        else:
            att_elo = def_elo = 1500.0

        last_date = last['Date']

        pts, scored, conceded, opp_elos = [], [], [], []
        sot, ht_pairs, corners, fouls, cards = [], [], [], [], []
        home_scored, home_conc, home_pts = [], [], []
        away_scored, away_conc, away_pts = [], [], []

        has_sot = 'HST' in tm.columns and 'AST' in tm.columns
        has_ht = 'HTHG' in tm.columns and 'HTAG' in tm.columns
        has_corners = 'HC' in tm.columns and 'AC' in tm.columns
        has_fouls = 'HF' in tm.columns and 'AF' in tm.columns
        has_cards = 'HY' in tm.columns and 'AY' in tm.columns

        for _, r in tm.iterrows():
            if r['HomeTeam'] == team:
                p = 3 if r['FTR'] == 'H' else 1 if r['FTR'] == 'D' else 0
                pts.append(p)
                scored.append(r['FTHG'])
                conceded.append(r['FTAG'])
                oe = (_safe(r.get('AwayAttackElo'), 1500) +
                      _safe(r.get('AwayDefenseElo'), 1500)) / 2.0
                opp_elos.append(oe)
                home_scored.append(r['FTHG'])
                home_conc.append(r['FTAG'])
                home_pts.append(p)
                if has_sot and pd.notna(r.get('HST')):
                    sot.append(r['HST'])
                if has_ht and pd.notna(r.get('HTHG')):
                    ht_pairs.append((r['HTHG'], r['FTHG']))
                if has_corners and pd.notna(r.get('HC')):
                    corners.append(r['HC'])
                if has_fouls and pd.notna(r.get('HF')):
                    fouls.append(r['HF'])
                if has_cards:
                    hy = _safe(r.get('HY'), 0)
                    hr = _safe(r.get('HR'), 0)
                    if hy or hr:
                        cards.append(hy + hr * 2)
            else:
                p = 3 if r['FTR'] == 'A' else 1 if r['FTR'] == 'D' else 0
                pts.append(p)
                scored.append(r['FTAG'])
                conceded.append(r['FTHG'])
                oe = (_safe(r.get('HomeAttackElo'), 1500) +
                      _safe(r.get('HomeDefenseElo'), 1500)) / 2.0
                opp_elos.append(oe)
                away_scored.append(r['FTAG'])
                away_conc.append(r['FTHG'])
                away_pts.append(p)
                if has_sot and pd.notna(r.get('AST')):
                    sot.append(r['AST'])
                if has_ht and pd.notna(r.get('HTAG')):
                    ht_pairs.append((r['HTAG'], r['FTAG']))
                if has_corners and pd.notna(r.get('AC')):
                    corners.append(r['AC'])
                if has_fouls and pd.notna(r.get('AF')):
                    fouls.append(r['AF'])
                if has_cards:
                    ay = _safe(r.get('AY'), 0)
                    ar = _safe(r.get('AR'), 0)
                    if ay or ar:
                        cards.append(ay + ar * 2)

        form = sum(pts[-fw:])
        momentum = _form_slope(pts, fw)

        avg_scored = np.mean(scored[-window:]) if scored else 0.0
        avg_conceded = np.mean(conceded[-window:]) if conceded else 0.0
        gd_list = [s - c for s, c in zip(scored[-window:], conceded[-window:])]
        avg_gd = np.mean(gd_list) if gd_list else 0.0

        avg_scored_3 = np.mean(scored[-sw:]) if scored else 0.0
        avg_conceded_3 = np.mean(conceded[-sw:]) if conceded else 0.0
        gd3 = [s - c for s, c in zip(scored[-sw:], conceded[-sw:])]
        avg_gd_3 = np.mean(gd3) if gd3 else 0.0

        n_sp = min(len(scored), window)
        if n_sp > 0:
            sc_w, cc_w = scored[-window:], conceded[-window:]
            cs_pct = sum(1 for c in cc_w if c == 0) / n_sp
            fts_pct = sum(1 for s in sc_w if s == 0) / n_sp
            btts_pct = sum(1 for s, c in zip(sc_w, cc_w) if s > 0 and c > 0) / n_sp
        else:
            cs_pct = fts_pct = btts_pct = 0.0

        sos = float(np.mean(opp_elos[-window:])) if opp_elos else float(config.ELO_INITIAL)

        goals_at_home = np.mean(home_scored[-window:]) if home_scored else 0.0
        conceded_at_home = np.mean(home_conc[-window:]) if home_conc else 0.0
        goals_away = np.mean(away_scored[-window:]) if away_scored else 0.0
        conceded_away = np.mean(away_conc[-window:]) if away_conc else 0.0
        form_home = sum(home_pts[-fw:]) if home_pts else 0
        form_away = sum(away_pts[-fw:]) if away_pts else 0

        lt = league_tables.get(league, {}).get(team, {})
        league_pos = lt.get('norm_pos', 0.5)
        ppg = lt.get('ppg', 0.0)

        avg_sot = np.mean(sot[-window:]) if sot else np.nan
        if ht_pairs:
            rec = ht_pairs[-window:]
            t_ft = sum(ft for _, ft in rec)
            t_ht = sum(ht for ht, _ in rec)
            ht_ratio = t_ht / t_ft if t_ft > 0 else 0.5
        else:
            ht_ratio = np.nan
        avg_corners = np.mean(corners[-window:]) if corners else np.nan
        avg_fouls = np.mean(fouls[-window:]) if fouls else np.nan
        avg_cards = np.mean(cards[-window:]) if cards else np.nan

        stats.append({
            'Team': team,
            'FinalAttackElo': att_elo,
            'FinalDefenseElo': def_elo,
            'FinalForm': form,
            'Momentum': momentum,
            'RollingGoalsScored': avg_scored,
            'RollingGoalsConceded': avg_conceded,
            'GoalDiff': avg_gd,
            'RollingGoalsScored_3': avg_scored_3,
            'RollingGoalsConceded_3': avg_conceded_3,
            'GoalDiff_3': avg_gd_3,
            'CleanSheetPct': cs_pct,
            'FailToScorePct': fts_pct,
            'BTTSPct': btts_pct,
            'LeaguePos': league_pos,
            'PointsPerGame': ppg,
            'GoalsAtHome': goals_at_home,
            'ConcededAtHome': conceded_at_home,
            'GoalsAway': goals_away,
            'ConcededAway': conceded_away,
            'FormHome': form_home,
            'FormAway': form_away,
            'SOS': sos,
            'RollingSoT': avg_sot,
            'HTGoalRatio': ht_ratio,
            'RollingCorners': avg_corners,
            'RollingFouls': avg_fouls,
            'RollingCards': avg_cards,
            'LastMatchDate': last_date,
            'League': league,
        })

    return pd.DataFrame(stats).set_index('Team')


def _safe(val, default):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return float(val)


def _form_slope(history: list, window: int) -> float:
    recent = history[-window:]
    n = len(recent)
    if n < 3:
        return 0.0
    x = np.arange(n, dtype=float)
    y = np.array(recent, dtype=float)
    xm = x.mean()
    ym = y.mean()
    denom = ((x - xm) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - xm) * (y - ym)).sum() / denom)


# =============================================================================
# HEAD-TO-HEAD (prediction-time)
# =============================================================================
def compute_h2h_stats(df: pd.DataFrame, home_team: str, away_team: str,
                      n_meetings: int = None) -> dict:
    """Return H2H stats dict: gd, home_win_rate, avg_total_goals."""
    if n_meetings is None:
        n_meetings = config.H2H_WINDOW
    meetings = df[
        ((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)) |
        ((df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team))
    ].sort_values('Date')

    if meetings.empty:
        return {'gd': 0.0, 'home_win_rate': 0.0, 'avg_total_goals': 0.0}

    recent = meetings.tail(n_meetings)
    gds, totals, wins = [], [], 0
    for _, r in recent.iterrows():
        fthg, ftag = int(r['FTHG']), int(r['FTAG'])
        totals.append(fthg + ftag)
        if r['HomeTeam'] == home_team:
            gds.append(fthg - ftag)
            if fthg > ftag:
                wins += 1
        else:
            gds.append(ftag - fthg)
            if ftag > fthg:
                wins += 1

    return {
        'gd': float(np.mean(gds)),
        'home_win_rate': wins / len(recent),
        'avg_total_goals': float(np.mean(totals)),
    }


def compute_h2h_gd(df: pd.DataFrame, home_team: str, away_team: str,
                    n_meetings: int = None) -> float:
    """Backward-compatible wrapper."""
    return compute_h2h_stats(df, home_team, away_team, n_meetings)['gd']


# =============================================================================
# FIXTURE FEATURE VECTOR
# =============================================================================
def create_fixture_features(home_stats: pd.Series,
                            away_stats: pd.Series,
                            h2h: dict | float,
                            league: str,
                            prediction_date=None,
                            historical_data: pd.DataFrame = None,
                            **_kwargs) -> pd.DataFrame:
    """Build a single-row DataFrame with all features for prediction."""
    is_rich = config.LEAGUE_REGISTRY.get(league, {}).get('type') == 'rich'

    if isinstance(h2h, (int, float)):
        h2h = {'gd': float(h2h), 'home_win_rate': 0.0, 'avg_total_goals': 0.0}

    if prediction_date is not None:
        h_last = home_stats.get('LastMatchDate')
        a_last = away_stats.get('LastMatchDate')
        h_rest = min((prediction_date - h_last).days, 30.0) if pd.notna(h_last) else 14.0
        a_rest = min((prediction_date - a_last).days, 30.0) if pd.notna(a_last) else 14.0
    else:
        h_rest = 7.0
        a_rest = 7.0

    if historical_data is not None:
        hw = getattr(config, 'LEAGUE_HOME_WIN_WINDOW', 150)
        ldf = historical_data[historical_data['league'] == league].sort_values('Date').tail(hw)
        league_hr = float((ldf['FTR'] == 'H').mean()) if len(ldf) > 0 else 0.0
    else:
        league_hr = 0.45

    row = {
        'AttackElo_diff': home_stats['FinalAttackElo'] - away_stats['FinalAttackElo'],
        'DefenseElo_diff': home_stats['FinalDefenseElo'] - away_stats['FinalDefenseElo'],
        'HomeForm': home_stats['FinalForm'],
        'AwayForm': away_stats['FinalForm'],
        'FormDiff': home_stats['FinalForm'] - away_stats['FinalForm'],
        'HomeMomentum': home_stats.get('Momentum', 0.0),
        'AwayMomentum': away_stats.get('Momentum', 0.0),
        'RollingHomeGoals': home_stats['RollingGoalsScored'],
        'RollingAwayGoals': away_stats['RollingGoalsScored'],
        'RollingHomeConceded': home_stats['RollingGoalsConceded'],
        'RollingAwayConceded': away_stats['RollingGoalsConceded'],
        'GoalDiff_Home': home_stats['GoalDiff'],
        'GoalDiff_Away': away_stats['GoalDiff'],
        'RollingHomeGoals_3': home_stats.get('RollingGoalsScored_3', 0.0),
        'RollingAwayGoals_3': away_stats.get('RollingGoalsScored_3', 0.0),
        'RollingHomeConceded_3': home_stats.get('RollingGoalsConceded_3', 0.0),
        'RollingAwayConceded_3': away_stats.get('RollingGoalsConceded_3', 0.0),
        'GoalDiff_Home_3': home_stats.get('GoalDiff_3', 0.0),
        'GoalDiff_Away_3': away_stats.get('GoalDiff_3', 0.0),
        'H2H_GD': h2h['gd'],
        'H2H_HomeWinRate': h2h['home_win_rate'],
        'H2H_AvgTotalGoals': h2h['avg_total_goals'],
        'HomeDaysRest': h_rest,
        'AwayDaysRest': a_rest,
        'RestAdvantage': h_rest - a_rest,
        'HomeCleanSheetPct': home_stats.get('CleanSheetPct', 0.0),
        'AwayCleanSheetPct': away_stats.get('CleanSheetPct', 0.0),
        'HomeFailToScorePct': home_stats.get('FailToScorePct', 0.0),
        'AwayFailToScorePct': away_stats.get('FailToScorePct', 0.0),
        'HomeBTTSPct': home_stats.get('BTTSPct', 0.0),
        'AwayBTTSPct': away_stats.get('BTTSPct', 0.0),
        'HomeLeaguePos': home_stats.get('LeaguePos', 0.5),
        'AwayLeaguePos': away_stats.get('LeaguePos', 0.5),
        'LeaguePosDiff': home_stats.get('LeaguePos', 0.5) - away_stats.get('LeaguePos', 0.5),
        'HomePointsPerGame': home_stats.get('PointsPerGame', 0.0),
        'AwayPointsPerGame': away_stats.get('PointsPerGame', 0.0),
        'HomeGoalsAtHome': home_stats.get('GoalsAtHome', 0.0),
        'AwayGoalsAway': away_stats.get('GoalsAway', 0.0),
        'HomeConcededAtHome': home_stats.get('ConcededAtHome', 0.0),
        'AwayConcededAway': away_stats.get('ConcededAway', 0.0),
        'HomeFormHome': home_stats.get('FormHome', 0),
        'AwayFormAway': away_stats.get('FormAway', 0),
        'HomeSOS': home_stats.get('SOS', float(config.ELO_INITIAL)),
        'AwaySOS': away_stats.get('SOS', float(config.ELO_INITIAL)),
        'LeagueHomeWinRate': league_hr,
    }

    if is_rich:
        row['RollingHomeSoT'] = _safe(home_stats.get('RollingSoT'), 0.0)
        row['RollingAwaySoT'] = _safe(away_stats.get('RollingSoT'), 0.0)
        row['HTGoalRatio_Home'] = _safe(home_stats.get('HTGoalRatio'), 0.5)
        row['HTGoalRatio_Away'] = _safe(away_stats.get('HTGoalRatio'), 0.5)
        row['RollingHomeCorners'] = _safe(home_stats.get('RollingCorners'), 0.0)
        row['RollingAwayCorners'] = _safe(away_stats.get('RollingCorners'), 0.0)
        row['CornerDiff'] = row['RollingHomeCorners'] - row['RollingAwayCorners']
        row['RollingHomeFouls'] = _safe(home_stats.get('RollingFouls'), 0.0)
        row['RollingAwayFouls'] = _safe(away_stats.get('RollingFouls'), 0.0)
        row['RollingHomeCards'] = _safe(home_stats.get('RollingCards'), 0.0)
        row['RollingAwayCards'] = _safe(away_stats.get('RollingCards'), 0.0)

    return pd.DataFrame([row])


# =============================================================================
# MODEL I/O
# =============================================================================
def load_model(model_type: str, league: str, tier: int = 1,
               model_set: str = 'current', engine: str = 'xgb'):
    path = _model_path(model_type, league, tier, model_set, engine)
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def save_model(model, model_type: str, league: str, tier: int = 1,
               model_set: str = 'current', engine: str = 'xgb'):
    path = _model_path(model_type, league, tier, model_set, engine)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def _model_path(model_type, league, tier, model_set, engine):
    tier_dir = config.get_tier_dir(tier=tier, model_set=model_set)
    if engine == 'xgb':
        filename = f"model_{model_type}_{league}.joblib"
    else:
        filename = f"model_{model_type}_{league}_{engine}.joblib"
    return os.path.join(tier_dir, filename)


# =============================================================================
# HELPERS
# =============================================================================
def get_available_leagues(tier: int = 1, model_set: str = 'current') -> list:
    tier_dir = config.get_tier_dir(tier=tier, model_set=model_set)
    if not os.path.isdir(tier_dir):
        return []
    files = glob.glob(os.path.join(tier_dir, 'model_1x2_*.joblib'))
    leagues = {
        os.path.basename(f).replace('model_1x2_', '').replace('.joblib', '')
        for f in files
    }
    return sorted(l for l in leagues if not l.endswith('_lgbm'))


def get_all_teams(df: pd.DataFrame) -> list:
    return sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())


def safe_odds(probability: float) -> float:
    return 1 / probability if probability > 0 else float('inf')


def confidence_label(prob: float) -> str:
    if prob >= 0.60:
        return 'Strong'
    elif prob >= 0.50:
        return 'Moderate'
    return 'Low'


def format_prediction_table(predictions: dict, **_kwargs) -> str:
    lines = [
        "=" * 60,
        f"  {predictions['home_team']} vs {predictions['away_team']}",
        f"  League: {predictions['league']}",
        "=" * 60, "",
    ]

    p = predictions.get('1x2', {})
    if p:
        top_p = max(p['home'], p['draw'], p['away'])
        conf = confidence_label(top_p)
        anchored = ' (market-anchored)' if 'market' in predictions else ''
        lines.append(f"1X2 Market:  [{conf}]{anchored}")
        lines.append(f"  Home: {p['home']:.1%}  (@ {safe_odds(p['home']):.2f})")
        lines.append(f"  Draw: {p['draw']:.1%}  (@ {safe_odds(p['draw']):.2f})")
        lines.append(f"  Away: {p['away']:.1%}  (@ {safe_odds(p['away']):.2f})")
        mkt = predictions.get('market')
        pm = predictions.get('1x2_model')
        if mkt and pm:
            o = mkt['odds']
            lines.append(f"  Book: {o['home']:.2f} / {o['draw']:.2f} / "
                         f"{o['away']:.2f}   Model alone: {pm['home']:.0%} / "
                         f"{pm['draw']:.0%} / {pm['away']:.0%}")
        lines += ["", "Double Chance:"]
        lines.append(f"  1X: {p['home'] + p['draw']:.1%}")
        lines.append(f"  X2: {p['draw'] + p['away']:.1%}")
        lines.append(f"  12: {p['home'] + p['away']:.1%}")
        lines.append("")

    ou15 = predictions.get('ou15')
    ou25 = predictions.get('ou25')
    ou35 = predictions.get('ou35')
    if ou15 and ou25 and ou35:
        p15, p25, p35 = ou15['over'], ou25['over'], ou35['over']
        p25 = min(p25, p15)
        p35 = min(p35, p25)
        lines.append("Goals Ladder:")
        lines.append(f"  0-1 goals: {1 - p15:>6.1%}")
        lines.append(f"  2 goals:   {p15 - p25:>6.1%}")
        lines.append(f"  3 goals:   {p25 - p35:>6.1%}")
        lines.append(f"  4+ goals:  {p35:>6.1%}")
        lines.append("")

    if ou25:
        conf = confidence_label(max(ou25['over'], ou25['under']))
        lines.append(f"Over/Under 2.5 Goals:  [{conf}]")
        lines.append(f"  Over:  {ou25['over']:.1%}  (@ {safe_odds(ou25['over']):.2f})")
        lines.append(f"  Under: {ou25['under']:.1%}  (@ {safe_odds(ou25['under']):.2f})")
        lines.append("")

    if ou15:
        lines.append("Over/Under 1.5 Goals:")
        lines.append(f"  Over:  {ou15['over']:.1%}  (@ {safe_odds(ou15['over']):.2f})")
        lines.append(f"  Under: {ou15['under']:.1%}  (@ {safe_odds(ou15['under']):.2f})")
        lines.append("")

    if ou35:
        lines.append("Over/Under 3.5 Goals:")
        lines.append(f"  Over:  {ou35['over']:.1%}  (@ {safe_odds(ou35['over']):.2f})")
        lines.append(f"  Under: {ou35['under']:.1%}  (@ {safe_odds(ou35['under']):.2f})")
        lines.append("")

    btts = predictions.get('btts')
    if btts:
        conf = confidence_label(max(btts['yes'], btts['no']))
        lines.append(f"Both Teams To Score:  [{conf}]")
        lines.append(f"  Yes: {btts['yes']:.1%}  (@ {safe_odds(btts['yes']):.2f})")
        lines.append(f"  No:  {btts['no']:.1%}  (@ {safe_odds(btts['no']):.2f})")
        lines.append("")

    ht1x2 = predictions.get('ht1x2')
    if ht1x2:
        lines.append("Half-Time Result:")
        lines.append(f"  Home: {ht1x2['home']:.1%}  Draw: {ht1x2['draw']:.1%}  Away: {ht1x2['away']:.1%}")
        lines.append("")

    htou = predictions.get('htou05')
    if htou:
        lines.append("HT Over/Under 0.5:")
        lines.append(f"  Over:  {htou['over']:.1%}  Under: {htou['under']:.1%}")
        lines.append("")

    xg = predictions.get('xg')
    if xg:
        lines.append(f"Expected Goals: {xg['home']:.2f} - {xg['away']:.2f}")
        lines.append("")
        lines.append("Most Likely Scores:")
        for score, prob in predictions.get('top_scores', []):
            lines.append(f"  {score}: {prob:.1%}")
        poisson_data = predictions.get('poisson')
        if poisson_data:
            lines += ["", "Poisson Cross-Check:"]
            lines.append(f"  O/U 2.5: Over {poisson_data['over25']:.1%} / Under {poisson_data['under25']:.1%}")
            lines.append(f"  BTTS: Yes {poisson_data['btts_yes']:.1%} / No {poisson_data['btts_no']:.1%}")

    lines.append("=" * 60)
    return "\n".join(lines)
