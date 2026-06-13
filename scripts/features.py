"""
Feature engineering: Elo (attack/defense), form, rest, rolling stats,
scoring patterns, H2H, league table context, venue splits, SOS,
league home-win rate, and targets. All strictly causal (pre-match only).
"""

import numpy as np; import pandas as pd; from . import config


def expected_outcome(rating1: float, rating2: float) -> float:
    return 1 / (1 + 10 ** ((rating2 - rating1) / 400))


def calculate_elo_for_league(
    df: pd.DataFrame,
    initial_elo: int = None,
    k: int = None,
    season_regression: float = None,
    margin_factor: float = None,
) -> pd.DataFrame:
    if initial_elo is None:
        initial_elo = config.ELO_INITIAL
    if k is None:
        k = config.ELO_K_FACTOR
    if season_regression is None:
        season_regression = config.ELO_SEASON_REGRESSION
    if margin_factor is None:
        margin_factor = config.ELO_MARGIN_FACTOR

    league_mean = float(initial_elo)
    att_elo: dict[str, float] = {}
    def_elo: dict[str, float] = {}
    last_played: dict[str, pd.Timestamp] = {}

    h_att, h_def, a_att, a_def = [], [], [], []

    for _, row in df.iterrows():
        ht, at = row['HomeTeam'], row['AwayTeam']
        date = row['Date']

        for team in (ht, at):
            prev = last_played.get(team)
            if prev is not None and season_regression > 0:
                gap = (date - prev).days
                if gap > 60:
                    a0 = att_elo.get(team, initial_elo)
                    d0 = def_elo.get(team, initial_elo)
                    att_elo[team] = (1.0 - season_regression) * a0 + season_regression * league_mean
                    def_elo[team] = (1.0 - season_regression) * d0 + season_regression * league_mean

        ha = att_elo.get(ht, initial_elo)
        hd = def_elo.get(ht, initial_elo)
        aa = att_elo.get(at, initial_elo)
        ad = def_elo.get(at, initial_elo)

        h_att.append(ha)
        h_def.append(hd)
        a_att.append(aa)
        a_def.append(ad)

        fthg, ftag = row['FTHG'], row['FTAG']
        gd_abs = abs(int(fthg) - int(ftag))
        k_eff = k * (1.0 + margin_factor * gd_abs)

        e_home_att = expected_outcome(ha, ad)
        e_away_att = expected_outcome(aa, hd)

        total = fthg + ftag if (fthg + ftag) > 0 else 1
        s_home_att = fthg / total
        s_away_att = ftag / total

        att_elo[ht] = ha + k_eff * (s_home_att - e_home_att)
        att_elo[at] = aa + k_eff * (s_away_att - e_away_att)

        e_home_def = expected_outcome(hd, aa)
        e_away_def = expected_outcome(ad, ha)

        s_home_def = 1 - (ftag / total)
        s_away_def = 1 - (fthg / total)

        def_elo[ht] = hd + k_eff * (s_home_def - e_home_def)
        def_elo[at] = ad + k_eff * (s_away_def - e_away_def)

        last_played[ht] = date
        last_played[at] = date

    df = df.copy()
    df['HomeAttackElo'] = h_att
    df['HomeDefenseElo'] = h_def
    df['AwayAttackElo'] = a_att
    df['AwayDefenseElo'] = a_def

    df['FinalHomeAttackElo'] = df['HomeTeam'].map(att_elo)
    df['FinalHomeDefenseElo'] = df['HomeTeam'].map(def_elo)
    df['FinalAwayAttackElo'] = df['AwayTeam'].map(att_elo)
    df['FinalAwayDefenseElo'] = df['AwayTeam'].map(def_elo)
    return df


def get_form_points(df: pd.DataFrame, n_matches: int = None) -> pd.DataFrame:
    if n_matches is None:
        n_matches = config.FORM_WINDOW

    team_history: dict[str, list[int]] = {}
    home_form, away_form = [], []
    home_momentum, away_momentum = [], []

    for _, row in df.iterrows():
        ht, at = row['HomeTeam'], row['AwayTeam']

        h_hist = team_history.get(ht, [])
        a_hist = team_history.get(at, [])
        home_form.append(sum(h_hist[-n_matches:]) if h_hist else 0)
        away_form.append(sum(a_hist[-n_matches:]) if a_hist else 0)

        home_momentum.append(_form_slope(h_hist, n_matches))
        away_momentum.append(_form_slope(a_hist, n_matches))

        res = row['FTR']
        hp = 3 if res == 'H' else 1 if res == 'D' else 0
        ap = 3 if res == 'A' else 1 if res == 'D' else 0
        team_history.setdefault(ht, []).append(hp)
        team_history.setdefault(at, []).append(ap)

    df = df.copy()
    df['HomeForm'] = home_form
    df['AwayForm'] = away_form
    df['HomeMomentum'] = home_momentum
    df['AwayMomentum'] = away_momentum
    return df


def _form_slope(history: list, window: int) -> float:
    recent = history[-window:]
    n = len(recent)
    if n < 3:
        return 0.0
    x = np.arange(n, dtype=float)
    y = np.array(recent, dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


def compute_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    last_match: dict[str, pd.Timestamp] = {}
    home_rest, away_rest = [], []

    for _, row in df.iterrows():
        ht, at = row['HomeTeam'], row['AwayTeam']
        date = row['Date']

        h_last = last_match.get(ht)
        a_last = last_match.get(at)
        h_days = (date - h_last).days if h_last is not None else 14.0
        a_days = (date - a_last).days if a_last is not None else 14.0
        home_rest.append(min(h_days, 30.0))
        away_rest.append(min(a_days, 30.0))

        last_match[ht] = date
        last_match[at] = date

    df = df.copy()
    df['HomeDaysRest'] = home_rest
    df['AwayDaysRest'] = away_rest
    df['RestAdvantage'] = df['HomeDaysRest'] - df['AwayDaysRest']
    return df


def compute_rolling_stats(
    df: pd.DataFrame, is_rich: bool = False, window: int = None
) -> pd.DataFrame:
    if window is None:
        window = config.ROLLING_WINDOW
    sw = config.SHORT_WINDOW

    team_scored: dict[str, list] = {}
    team_conceded: dict[str, list] = {}
    team_sot: dict[str, list] = {}
    team_ht: dict[str, list] = {}
    team_corners: dict[str, list] = {}
    team_fouls: dict[str, list] = {}
    team_cards: dict[str, list] = {}

    r_hg, r_ag, r_hc, r_ac = [], [], [], []
    gd_h, gd_a = [], []
    r_hg3, r_ag3, r_hc3, r_ac3 = [], [], [], []
    gd_h3, gd_a3 = [], []

    r_hsot, r_asot = [], []
    ht_r_h, ht_r_a = [], []
    r_hcorn, r_acorn, corn_diff = [], [], []
    r_hfoul, r_afoul = [], []
    r_hcard, r_acard = [], []

    has_sot = is_rich and 'HST' in df.columns and 'AST' in df.columns
    has_ht = is_rich and 'HTHG' in df.columns and 'HTAG' in df.columns
    has_corners = is_rich and 'HC' in df.columns and 'AC' in df.columns
    has_fouls = is_rich and 'HF' in df.columns and 'AF' in df.columns
    has_cards = is_rich and 'HY' in df.columns and 'AY' in df.columns

    for _, row in df.iterrows():
        ht, at = row['HomeTeam'], row['AwayTeam']

        hs = team_scored.get(ht, [])
        hc = team_conceded.get(ht, [])
        as_ = team_scored.get(at, [])
        ac = team_conceded.get(at, [])

        r_hg.append(np.mean(hs[-window:]) if hs else 0.0)
        r_ag.append(np.mean(as_[-window:]) if as_ else 0.0)
        r_hc.append(np.mean(hc[-window:]) if hc else 0.0)
        r_ac.append(np.mean(ac[-window:]) if ac else 0.0)

        hgd = [s - c for s, c in zip(hs[-window:], hc[-window:])]
        agd = [s - c for s, c in zip(as_[-window:], ac[-window:])]
        gd_h.append(np.mean(hgd) if hgd else 0.0)
        gd_a.append(np.mean(agd) if agd else 0.0)

        r_hg3.append(np.mean(hs[-sw:]) if hs else 0.0)
        r_ag3.append(np.mean(as_[-sw:]) if as_ else 0.0)
        r_hc3.append(np.mean(hc[-sw:]) if hc else 0.0)
        r_ac3.append(np.mean(ac[-sw:]) if ac else 0.0)
        hgd3 = [s - c for s, c in zip(hs[-sw:], hc[-sw:])]
        agd3 = [s - c for s, c in zip(as_[-sw:], ac[-sw:])]
        gd_h3.append(np.mean(hgd3) if hgd3 else 0.0)
        gd_a3.append(np.mean(agd3) if agd3 else 0.0)

        if has_sot:
            h_s = team_sot.get(ht, [])
            a_s = team_sot.get(at, [])
            r_hsot.append(np.mean(h_s[-window:]) if h_s else 0.0)
            r_asot.append(np.mean(a_s[-window:]) if a_s else 0.0)

        if has_ht:
            h_ht = team_ht.get(ht, [])
            a_ht = team_ht.get(at, [])
            ht_r_h.append(_ht_ratio(h_ht[-window:]))
            ht_r_a.append(_ht_ratio(a_ht[-window:]))

        if has_corners:
            hcl = team_corners.get(ht, [])
            acl = team_corners.get(at, [])
            hcm = np.mean(hcl[-window:]) if hcl else 0.0
            acm = np.mean(acl[-window:]) if acl else 0.0
            r_hcorn.append(hcm)
            r_acorn.append(acm)
            corn_diff.append(hcm - acm)

        if has_fouls:
            hfl = team_fouls.get(ht, [])
            afl = team_fouls.get(at, [])
            r_hfoul.append(np.mean(hfl[-window:]) if hfl else 0.0)
            r_afoul.append(np.mean(afl[-window:]) if afl else 0.0)

        if has_cards:
            hcrdl = team_cards.get(ht, [])
            acrdl = team_cards.get(at, [])
            r_hcard.append(np.mean(hcrdl[-window:]) if hcrdl else 0.0)
            r_acard.append(np.mean(acrdl[-window:]) if acrdl else 0.0)

        fthg, ftag = row['FTHG'], row['FTAG']
        team_scored.setdefault(ht, []).append(fthg)
        team_conceded.setdefault(ht, []).append(ftag)
        team_scored.setdefault(at, []).append(ftag)
        team_conceded.setdefault(at, []).append(fthg)

        if has_sot:
            team_sot.setdefault(ht, []).append(
                row['HST'] if pd.notna(row.get('HST')) else 0)
            team_sot.setdefault(at, []).append(
                row['AST'] if pd.notna(row.get('AST')) else 0)

        if has_ht:
            hthg = row['HTHG'] if pd.notna(row.get('HTHG')) else 0
            htag = row['HTAG'] if pd.notna(row.get('HTAG')) else 0
            team_ht.setdefault(ht, []).append((hthg, fthg))
            team_ht.setdefault(at, []).append((htag, ftag))

        if has_corners:
            team_corners.setdefault(ht, []).append(
                row['HC'] if pd.notna(row.get('HC')) else 0)
            team_corners.setdefault(at, []).append(
                row['AC'] if pd.notna(row.get('AC')) else 0)

        if has_fouls:
            team_fouls.setdefault(ht, []).append(
                row['HF'] if pd.notna(row.get('HF')) else 0)
            team_fouls.setdefault(at, []).append(
                row['AF'] if pd.notna(row.get('AF')) else 0)

        if has_cards:
            hy = row['HY'] if pd.notna(row.get('HY')) else 0
            hr = row['HR'] if pd.notna(row.get('HR')) else 0
            ay = row['AY'] if pd.notna(row.get('AY')) else 0
            ar = row['AR'] if pd.notna(row.get('AR')) else 0
            team_cards.setdefault(ht, []).append(hy + hr * 2)
            team_cards.setdefault(at, []).append(ay + ar * 2)

    df = df.copy()
    df['RollingHomeGoals'] = r_hg
    df['RollingAwayGoals'] = r_ag
    df['RollingHomeConceded'] = r_hc
    df['RollingAwayConceded'] = r_ac
    df['GoalDiff_Home'] = gd_h
    df['GoalDiff_Away'] = gd_a
    df['RollingHomeGoals_3'] = r_hg3
    df['RollingAwayGoals_3'] = r_ag3
    df['RollingHomeConceded_3'] = r_hc3
    df['RollingAwayConceded_3'] = r_ac3
    df['GoalDiff_Home_3'] = gd_h3
    df['GoalDiff_Away_3'] = gd_a3

    if has_sot:
        df['RollingHomeSoT'] = r_hsot
        df['RollingAwaySoT'] = r_asot
    if has_ht:
        df['HTGoalRatio_Home'] = ht_r_h
        df['HTGoalRatio_Away'] = ht_r_a
    if has_corners:
        df['RollingHomeCorners'] = r_hcorn
        df['RollingAwayCorners'] = r_acorn
        df['CornerDiff'] = corn_diff
    if has_fouls:
        df['RollingHomeFouls'] = r_hfoul
        df['RollingAwayFouls'] = r_afoul
    if has_cards:
        df['RollingHomeCards'] = r_hcard
        df['RollingAwayCards'] = r_acard

    return df


def _ht_ratio(ht_pairs: list) -> float:
    if not ht_pairs:
        return 0.5
    total_ft = sum(ft for _, ft in ht_pairs)
    total_ht = sum(ht for ht, _ in ht_pairs)
    return total_ht / total_ft if total_ft > 0 else 0.5


def _season_key(d: pd.Timestamp) -> int:
    y = int(d.year)
    m = int(d.month)
    return y if m >= 8 else y - 1


def compute_scoring_patterns(df: pd.DataFrame, window: int = None) -> pd.DataFrame:
    if window is None:
        window = config.ROLLING_WINDOW

    team_scored: dict[str, list] = {}
    team_conceded: dict[str, list] = {}
    h_cs, a_cs = [], []
    h_fts, a_fts = [], []
    h_btts, a_btts = [], []

    for _, row in df.iterrows():
        ht, at = row['HomeTeam'], row['AwayTeam']

        hs, hc = team_scored.get(ht, []), team_conceded.get(ht, [])
        as_, ac = team_scored.get(at, []), team_conceded.get(at, [])

        def _pct(sc: list, cc: list, pred) -> float:
            n = min(len(sc), len(cc), window)
            if n == 0:
                return 0.0
            scw = sc[-window:]
            ccw = cc[-window:]
            hits = sum(1 for s, c in zip(scw, ccw) if pred(s, c))
            return hits / float(n)

        h_cs.append(_pct(hs, hc, lambda s, c: c == 0))
        a_cs.append(_pct(as_, ac, lambda s, c: c == 0))
        h_fts.append(_pct(hs, hc, lambda s, c: s == 0))
        a_fts.append(_pct(as_, ac, lambda s, c: s == 0))
        h_btts.append(_pct(hs, hc, lambda s, c: s > 0 and c > 0))
        a_btts.append(_pct(as_, ac, lambda s, c: s > 0 and c > 0))

        fthg, ftag = row['FTHG'], row['FTAG']
        team_scored.setdefault(ht, []).append(fthg)
        team_conceded.setdefault(ht, []).append(ftag)
        team_scored.setdefault(at, []).append(ftag)
        team_conceded.setdefault(at, []).append(fthg)

    df = df.copy()
    df['HomeCleanSheetPct'] = h_cs
    df['AwayCleanSheetPct'] = a_cs
    df['HomeFailToScorePct'] = h_fts
    df['AwayFailToScorePct'] = a_fts
    df['HomeBTTSPct'] = h_btts
    df['AwayBTTSPct'] = a_btts
    return df


def compute_h2h_features(df: pd.DataFrame, n_meetings: int = None) -> pd.DataFrame:
    if n_meetings is None:
        n_meetings = config.H2H_WINDOW

    h2h_hist: dict[tuple, list] = {}
    h2h_gd, h2h_wr, h2h_avg = [], [], []

    for _, row in df.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        key = tuple(sorted([home, away]))

        history = h2h_hist.get(key, [])
        recent = history[-n_meetings:]
        if recent:
            gds = []
            totals = []
            wins = 0
            for fh, fa, ghg, gag in recent:
                totals.append(ghg + gag)
                if fh == home:
                    gds.append(ghg - gag)
                    if ghg > gag:
                        wins += 1
                else:
                    gds.append(gag - ghg)
                    if gag > ghg:
                        wins += 1
            h2h_gd.append(float(np.mean(gds)))
            h2h_wr.append(wins / float(len(recent)))
            h2h_avg.append(float(np.mean(totals)))
        else:
            h2h_gd.append(0.0)
            h2h_wr.append(0.0)
            h2h_avg.append(0.0)

        fthg, ftag = int(row['FTHG']), int(row['FTAG'])
        h2h_hist.setdefault(key, []).append((home, away, fthg, ftag))

    df = df.copy()
    df['H2H_GD'] = h2h_gd
    df['H2H_HomeWinRate'] = h2h_wr
    df['H2H_AvgTotalGoals'] = h2h_avg
    return df


def compute_league_context(df: pd.DataFrame) -> pd.DataFrame:
    standings: dict[int, dict[str, dict]] = {}
    h_pos, a_pos = [], []
    h_ppg, a_ppg = [], []
    pos_diff = []

    for _, row in df.iterrows():
        date = row['Date']
        season = _season_key(date)
        ht, at = row['HomeTeam'], row['AwayTeam']
        table = standings.setdefault(season, {})

        def _team_row(team: str) -> dict:
            return table.setdefault(team, {'pts': 0, 'pld': 0, 'gf': 0, 'ga': 0})

        rh = _team_row(ht)
        ra = _team_row(at)

        def _norm_pos(team: str, tr: dict) -> float:
            if tr['pld'] < 3:
                return 0.5
            active = [(t, v) for t, v in table.items() if v['pld'] > 0]
            active.sort(
                key=lambda x: (-x[1]['pts'], -(x[1]['gf'] - x[1]['ga']), -x[1]['gf'])
            )
            ranks = {t: i + 1 for i, (t, _) in enumerate(active)}
            n = len(active)
            if n <= 1:
                return 0.5
            r = ranks.get(team, n)
            return (r - 1) / float(n - 1)

        def _ppg(tr: dict) -> float:
            return tr['pts'] / tr['pld'] if tr['pld'] > 0 else 0.0

        hp = _norm_pos(ht, rh)
        ap = _norm_pos(at, ra)
        h_pos.append(hp)
        a_pos.append(ap)
        h_ppg.append(_ppg(rh))
        a_ppg.append(_ppg(ra))
        pos_diff.append(hp - ap)

        res = row['FTR']
        fthg, ftag = int(row['FTHG']), int(row['FTAG'])
        rh['pld'] += 1
        ra['pld'] += 1
        rh['gf'] += fthg
        rh['ga'] += ftag
        ra['gf'] += ftag
        ra['ga'] += fthg
        if res == 'H':
            rh['pts'] += 3
        elif res == 'A':
            ra['pts'] += 3
        else:
            rh['pts'] += 1
            ra['pts'] += 1

    df = df.copy()
    df['HomeLeaguePos'] = h_pos
    df['AwayLeaguePos'] = a_pos
    df['LeaguePosDiff'] = pos_diff
    df['HomePointsPerGame'] = h_ppg
    df['AwayPointsPerGame'] = a_ppg
    return df


def compute_venue_stats(df: pd.DataFrame) -> pd.DataFrame:
    vw = config.ROLLING_WINDOW
    fw = config.FORM_WINDOW

    home_scored_h: dict[str, list] = {}
    home_conc_h: dict[str, list] = {}
    away_scored_a: dict[str, list] = {}
    away_conc_a: dict[str, list] = {}
    home_pts_h: dict[str, list] = {}
    away_pts_a: dict[str, list] = {}

    h_gh, a_ga, h_ch, a_ca = [], [], [], []
    h_fh, a_fa = [], []

    for _, row in df.iterrows():
        ht, at = row['HomeTeam'], row['AwayTeam']

        hgh = home_scored_h.get(ht, [])
        hch = home_conc_h.get(ht, [])
        asa = away_scored_a.get(at, [])
        aca = away_conc_a.get(at, [])
        hph = home_pts_h.get(ht, [])
        apa = away_pts_a.get(at, [])

        h_gh.append(np.mean(hgh[-vw:]) if hgh else 0.0)
        h_ch.append(np.mean(hch[-vw:]) if hch else 0.0)
        a_ga.append(np.mean(asa[-vw:]) if asa else 0.0)
        a_ca.append(np.mean(aca[-vw:]) if aca else 0.0)
        h_fh.append(sum(hph[-fw:]) if hph else 0)
        a_fa.append(sum(apa[-fw:]) if apa else 0)

        res = row['FTR']
        fthg, ftag = row['FTHG'], row['FTAG']
        hp = 3 if res == 'H' else 1 if res == 'D' else 0
        ap = 3 if res == 'A' else 1 if res == 'D' else 0

        home_scored_h.setdefault(ht, []).append(fthg)
        home_conc_h.setdefault(ht, []).append(ftag)
        away_scored_a.setdefault(at, []).append(ftag)
        away_conc_a.setdefault(at, []).append(fthg)
        home_pts_h.setdefault(ht, []).append(hp)
        away_pts_a.setdefault(at, []).append(ap)

    df = df.copy()
    df['HomeGoalsAtHome'] = h_gh
    df['AwayGoalsAway'] = a_ga
    df['HomeConcededAtHome'] = h_ch
    df['AwayConcededAway'] = a_ca
    df['HomeFormHome'] = h_fh
    df['AwayFormAway'] = a_fa
    return df


def compute_sos(df: pd.DataFrame, window: int = None) -> pd.DataFrame:
    if window is None:
        window = config.ROLLING_WINDOW

    home_opp: dict[str, list[float]] = {}
    away_opp: dict[str, list[float]] = {}
    h_sos, a_sos = [], []

    for _, row in df.iterrows():
        ht, at = row['HomeTeam'], row['AwayTeam']
        ho = home_opp.get(ht, [])
        ao_list = away_opp.get(at, [])

        h_slice = ho[-window:]
        a_slice = ao_list[-window:]
        init = float(config.ELO_INITIAL)
        h_sos.append(float(np.mean(h_slice)) if h_slice else init)
        a_sos.append(float(np.mean(a_slice)) if a_slice else init)

        oa = (float(row['AwayAttackElo']) + float(row['AwayDefenseElo'])) / 2.0
        oh = (float(row['HomeAttackElo']) + float(row['HomeDefenseElo'])) / 2.0
        home_opp.setdefault(ht, []).append(oa)
        away_opp.setdefault(at, []).append(oh)

    df = df.copy()
    df['HomeSOS'] = h_sos
    df['AwaySOS'] = a_sos
    return df


def compute_league_home_rate(df: pd.DataFrame, window: int = None) -> pd.DataFrame:
    if window is None:
        window = config.LEAGUE_HOME_WIN_WINDOW

    hist: list[int] = []
    rates = []

    for _, row in df.iterrows():
        if hist:
            w = hist[-window:]
            rates.append(float(np.mean(w)))
        else:
            rates.append(0.0)
        hist.append(1 if row['FTR'] == 'H' else 0)

    df = df.copy()
    df['LeagueHomeWinRate'] = rates
    return df


def create_features(df: pd.DataFrame, is_rich: bool = False) -> pd.DataFrame:
    df = df.copy()

    df['AttackElo_diff'] = df['HomeAttackElo'] - df['AwayAttackElo']
    df['DefenseElo_diff'] = df['HomeDefenseElo'] - df['AwayDefenseElo']

    df = get_form_points(df)
    df['FormDiff'] = df['HomeForm'] - df['AwayForm']

    df = compute_rest_days(df)

    df = compute_rolling_stats(df, is_rich=is_rich)

    df = compute_scoring_patterns(df)

    df = compute_h2h_features(df)

    df = compute_league_context(df)

    df = compute_venue_stats(df)

    df = compute_sos(df)

    df = compute_league_home_rate(df)

    total_goals = df['FTHG'] + df['FTAG']
    df['result_label'] = df['FTR'].map({'H': 2, 'D': 1, 'A': 0})
    df['over_2_5'] = (total_goals > 2.5).astype(int)
    df['over_1_5'] = (total_goals > 1.5).astype(int)
    df['over_3_5'] = (total_goals > 3.5).astype(int)
    df['btts'] = ((df['FTHG'] > 0) & (df['FTAG'] > 0)).astype(int)

    if is_rich and 'HTHG' in df.columns and 'HTAG' in df.columns:
        df['ht_result'] = df.get('HTR', pd.Series(dtype='object')).map(
            {'H': 2, 'D': 1, 'A': 0})
        hthg = pd.to_numeric(df['HTHG'], errors='coerce').fillna(0)
        htag = pd.to_numeric(df['HTAG'], errors='coerce').fillna(0)
        df['ht_over_0_5'] = ((hthg + htag) > 0.5).astype(int)

    df = df.dropna(subset=['result_label'])
    df['result_label'] = df['result_label'].astype(int)
    return df
