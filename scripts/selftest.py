#!/usr/bin/env python3
"""
Health check for the whole predictor
====================================
One command that answers "is anything quietly broken?" — run it after a data
refresh, a retrain, or any code change::

    ./venv/bin/python scripts/selftest.py
    ./venv/bin/python scripts/selftest.py --quick   # skip the slow app boot

Checks, in order of how expensive they are:

  1. Data integrity  — every league file really holds that league's matches.
     This is the check that would have caught English Conference results
     being served in answer to a Premier League request and trained as EPL.
  2. Freshness       — which leagues have gone stale (in season but no recent
     results), so a dead fetch cannot hide.
  3. Team leakage    — the same club appearing in two leagues.
  4. Toto maths      — Poisson-binomial against Monte Carlo, and the system
     optimiser against brute force.
  5. Models          — the saved ensembles load and predict.
  6. Internationals  — Elo + goal model fit and produce sane probabilities.
  7. App             — Streamlit boots and the Toto tab analyses a coupon
     (against a scratch coupon directory, never your real ones).

Exit code is 0 when everything passes, 1 if any check fails. Warnings (e.g. a
stale league) do not fail the run.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from scripts import config

PASS, FAIL, WARN = 'PASS', 'FAIL', 'WARN'
_results: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = '') -> None:
    _results.append((status, name, detail))
    icon = {PASS: '  ok  ', FAIL: ' FAIL ', WARN: ' warn '}[status]
    print(f"[{icon}] {name}" + (f" — {detail}" if detail else ''))


# ---------------------------------------------------------------- 1. integrity
def check_league_files() -> None:
    from scripts import fetch_latest
    bad, checked = [], 0
    for lg, src in config.FETCH_SOURCES.items():
        path = os.path.join(config.DATA_DIR, lg, src['target'])
        if not os.path.isfile(path):
            continue
        checked += 1
        with open(path, 'rb') as f:
            err = fetch_latest._validate_payload(f.read(), lg)
        if err:
            bad.append(f"{lg}:{src['target']} {err}")
    if bad:
        record(FAIL, 'league files hold the right league', '; '.join(bad))
    else:
        record(PASS, 'league files hold the right league',
               f'{checked} fetch targets verified')


def check_processed_leagues() -> pd.DataFrame:
    df = pd.read_csv(config.PROCESSED_DATA_FILE,
                     usecols=['Date', 'league', 'HomeTeam', 'AwayTeam'],
                     parse_dates=['Date'])
    known = set(config.LEAGUE_REGISTRY)
    unknown = sorted(set(df['league']) - known)
    if unknown:
        record(FAIL, 'processed data leagues are registered', str(unknown))
    else:
        record(PASS, 'processed data leagues are registered',
               f'{df["league"].nunique()} leagues, {len(df):,} matches')
    return df


# ---------------------------------------------------------------- 2. freshness
def check_freshness(df: pd.DataFrame, stale_days: int = 28) -> None:
    today = pd.Timestamp.today().normalize()
    last = df.groupby('league')['Date'].max()
    stale = {lg: (today - d).days for lg, d in last.items()
             if (today - d).days > stale_days}
    if stale:
        worst = sorted(stale.items(), key=lambda kv: -kv[1])
        record(WARN, 'league data freshness',
               'stale: ' + ', '.join(f'{lg} {d}d' for lg, d in worst[:6])
               + ('' if len(worst) <= 6 else f' (+{len(worst)-6} more)')
               + '  — off-season, or the season file is not published yet')
    else:
        record(PASS, 'league data freshness',
               f'all {len(last)} leagues within {stale_days}d')


def check_team_leakage(df: pd.DataFrame, window_days: int = 60) -> None:
    """A club in two leagues at the *same time* means a file went to the wrong
    folder. Across seasons it just means promotion or relegation, so only
    simultaneous overlap is suspicious."""
    recent = df[df['Date'] >= df['Date'].max() - pd.Timedelta(days=window_days)]
    pairs = {}
    for lg, sub in recent.groupby('league'):
        for t in set(sub['HomeTeam']) | set(sub['AwayTeam']):
            pairs.setdefault(t, set()).add(lg)
    shared = {t: sorted(lg) for t, lg in pairs.items() if len(lg) > 1}
    if shared:
        sample = list(itertools.islice(shared.items(), 5))
        record(FAIL, 'no club is in two leagues at once',
               f'{len(shared)} in the last {window_days}d: ' +
               ', '.join(f'{t}={lg}' for t, lg in sample))
    else:
        record(PASS, 'no club is in two leagues at once',
               f'{len(pairs)} clubs active in the last {window_days}d')


def check_config_wiring() -> None:
    """Every registered league must be wired everywhere it is consumed.

    These mappings live in different modules, and a league added to one but
    not another fails silently (its fixtures never map, its payloads are never
    validated). Austria/Poland/Romania were exactly this: registered and
    trained, but missing from the feed's country mapping.
    """
    from scripts import fixtures as fx
    problems = []
    for lg, info in config.LEAGUE_REGISTRY.items():
        if lg not in config.FETCH_SOURCES:
            problems.append(f'{lg}: no FETCH_SOURCES entry')
            continue
        if info['type'] == 'sparse':
            country = config.SPARSE_COUNTRY.get(lg)
            if not country:
                problems.append(f'{lg}: missing from SPARSE_COUNTRY '
                                '(payload validation disabled)')
            elif fx.COUNTRY_TO_LEAGUE.get(country) != lg:
                problems.append(f'{lg}: feed country mapping broken')
    extra = set(config.FETCH_SOURCES) - set(config.LEAGUE_REGISTRY)
    if extra:
        problems.append(f'FETCH_SOURCES has unregistered keys: {sorted(extra)}')
    if problems:
        record(FAIL, 'league config is fully wired', '; '.join(problems[:4]))
    else:
        record(PASS, 'league config is fully wired',
               f'{len(config.LEAGUE_REGISTRY)} leagues x fetch/validate/feed')


def check_name_resolution() -> None:
    """Aliases and fuzzy matching keep resolving what coupons rely on."""
    from scripts import toto
    cases = [('psg', 'Paris SG'), ('bvb', 'Dortmund'),
             ('gladbach', "M'gladbach"), ('atleti', 'Ath Madrid'),
             ('spurs', 'Tottenham'), ('forest', "Nott'm Forest")]
    bad = [f'{a}->{toto.apply_alias(a)}' for a, want in cases
           if toto.apply_alias(a) != want]
    ic = toto._load_intl()
    for typed, want in [('curacao', 'Curaçao'), ('usa', 'United States'),
                        ('turkiye', 'Turkey'), ('korea', 'South Korea')]:
        got = toto._intl_name(typed, ic)
        if got != want:
            bad.append(f'{typed}->{got}')
    if bad:
        record(FAIL, 'name aliases resolve', '; '.join(bad))
    else:
        record(PASS, 'name aliases resolve',
               f'{len(cases)} club + 4 national shorthands')


def check_tracker() -> None:
    """Save -> grade roundtrip against real recent results, in a temp dir."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        os.environ['FOOTBALL_PREDICTOR_TOTO_DIR'] = tmp
        try:
            from scripts import track
            df = pd.read_csv(config.PROCESSED_DATA_FILE,
                             usecols=['Date', 'HomeTeam', 'AwayTeam'],
                             parse_dates=['Date'])
            # Grading looks BACK_DAYS back from the save date (today), so the
            # sample must be that recent — not merely recent relative to the
            # data, which can lag.
            cutoff = (pd.Timestamp.today().normalize()
                      - pd.Timedelta(days=track.ResultLookup.BACK_DAYS - 1))
            recent = df[df['Date'] >= cutoff]
            if len(recent) < 4:
                record(WARN, 'coupon tracker roundtrip',
                       f'no matches since {cutoff.date()} to grade against '
                       '(data is stale)')
                return
            matches = [{'home': r.HomeTeam, 'away': r.AwayTeam, 'p1': 0.5,
                        'px': 0.3, 'p2': 0.2, 'pick': '1', 'src': 'model'}
                       for r in recent.head(4).itertuples()]
            track.save_coupon('turkish', matches, budget=1, threshold=2,
                              p_threshold=0.5, note='selftest')
            g = track.grade_all()[0]
            ok = g['graded'] == 4 and 0 <= g['correct'] <= 4
            record(PASS if ok else FAIL, 'coupon tracker roundtrip',
                   f"graded {g['graded']}/4, {g['correct']} correct")
        except Exception as e:
            record(FAIL, 'coupon tracker roundtrip',
                   f'{type(e).__name__}: {e}')
        finally:
            os.environ.pop('FOOTBALL_PREDICTOR_TOTO_DIR', None)


# --------------------------------------------------------------- 4. toto maths
def check_toto_math() -> None:
    from scripts import toto
    rng = np.random.default_rng(11)

    def rp(n):
        return [np.sort(rng.dirichlet([4.0, 3.0, 3.0]))[::-1] for _ in range(n)]

    # Poisson-binomial vs Monte Carlo
    sp = rp(9)
    cov = [1, 2, 3, 1, 2, 1, 1, 2, 1]
    qs = [float(np.sum(sp[i][:cov[i]])) for i in range(9)]
    analytic = toto.prob_at_least(qs, 6)
    sims = 200_000
    hits = np.zeros(sims, dtype=int)
    for i in range(9):
        p = np.asarray(sp[i], float); p = p / p.sum()
        hits += (rng.choice(3, size=sims, p=p) < cov[i]).astype(int)
    mc = (hits >= 6).mean()
    if abs(analytic - mc) < 0.005:
        record(PASS, 'Poisson-binomial matches Monte Carlo',
               f'{analytic:.4f} vs {mc:.4f}')
    else:
        record(FAIL, 'Poisson-binomial matches Monte Carlo',
               f'{analytic:.4f} vs {mc:.4f}')

    # optimiser vs brute force
    worst = 0.0
    for _ in range(4):
        n = int(rng.integers(5, 8))
        thr = n - int(rng.integers(1, 3))
        budget = int(rng.choice([8, 16, 24, 36]))
        sp = rp(n)
        _, _, got = toto.optimize_system(sp, thr, budget)
        best = 0.0
        for combo in itertools.product([1, 2, 3], repeat=n):
            if int(np.prod(combo)) > budget:
                continue
            q = [float(np.sum(sp[i][:combo[i]])) for i in range(n)]
            best = max(best, toto.prob_at_least(q, thr))
        worst = max(worst, best - got)
    if worst < 1e-6:
        record(PASS, 'system optimiser reaches the optimum',
               'exact on 4 brute-forced instances')
    else:
        record(FAIL, 'system optimiser reaches the optimum',
               f'shortfall {worst:.5f}')

    # parser round-trip
    txt = "Norway - Italy\nTurkey - Spain  1.95 3.40 3.90\nGalatasaray-Fenerbahce"
    got = toto.parse_lines(txt)
    ok = (len(got) == 3 and got.iloc[1]['o1'] == 1.95
          and got.iloc[2]['away'] == 'Fenerbahce')
    record(PASS if ok else FAIL, 'coupon parser',
           f'{len(got)} lines parsed' if ok else str(got.to_dict('records')))


# ------------------------------------------------------------------ 5. models
def check_models(quick: bool) -> None:
    """Every league with models must predict. Leagues that simply have not
    been trained yet are a warning, not a failure — but they must be named,
    because in the app they otherwise return nothing silently."""
    from scripts import data_loader, utils
    from scripts import predict as predict_mod
    t0 = time.time()
    hist = data_loader.load_processed_data()
    team_stats = utils.get_team_stats_table(hist)
    t2l = utils.get_team_to_league_map(hist)
    leagues = sorted({t2l[t] for t in t2l})
    try:
        have = set(utils.get_available_leagues())
    except Exception:
        have = set(leagues)

    untrained = sorted(set(leagues) - have)
    trained = [lg for lg in leagues if lg in have]
    if quick:
        trained = trained[:6]

    tried, ok, broken = 0, 0, []
    for lg in trained:
        teams = [t for t, l in t2l.items() if l == lg][:2]
        if len(teams) < 2:
            continue
        tried += 1
        try:
            p = predict_mod.predict_match(teams[0], teams[1], team_stats, t2l,
                                          hist, include_xg=False)
            probs = p.get('1x2')
            if probs and abs(sum(probs.values()) - 1.0) < 0.02:
                ok += 1
            else:
                broken.append(lg)
        except Exception as e:
            broken.append(f'{lg}({type(e).__name__})')
    record(PASS if not broken else FAIL, 'trained leagues predict',
           f'{ok}/{tried} in {time.time()-t0:.0f}s'
           + (f' — broken: {broken}' if broken else ''))
    if untrained:
        record(WARN, 'every league has models',
               f'{len(untrained)} untrained: ' + ', '.join(untrained[:8])
               + '  — they appear in the app but return no prediction')


# ---------------------------------------------------------- 6. internationals
def check_international() -> None:
    from scripts import international as intl
    try:
        df = intl.load_results()
        ratings, hist = intl.run_elo(df)
        model = intl.GoalModel().fit(hist)
        p = model.market_probs(ratings.get('Brazil', 1500),
                               ratings.get('Norway', 1500), neutral=True)
        tot = p['p_home'] + p['p_draw'] + p['p_away']
        sane = (abs(tot - 1.0) < 1e-6 and 0.15 < p['p_draw'] < 0.35
                and 0 < p['p_over25'] < 1)
        record(PASS if sane else FAIL, 'international model',
               f"{len(df):,} matches, draw={p['p_draw']:.1%}, "
               f"top Elo={max(ratings.values()):.0f}")
    except Exception as e:
        record(FAIL, 'international model', f'{type(e).__name__}: {e}')


# ---------------------------------------------------------------------- 7. app
def check_app() -> None:
    try:
        from streamlit.testing.v1 import AppTest
    except Exception as e:
        record(WARN, 'streamlit app boots', f'streamlit unavailable: {e}')
        return
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with tempfile.TemporaryDirectory() as tmp:
        # never touch the real saved coupons
        os.environ['FOOTBALL_PREDICTOR_TOTO_DIR'] = tmp
        try:
            at = AppTest.from_file(os.path.join(root, 'app.py'),
                                   default_timeout=600)
            at.run()
            if at.exception:
                record(FAIL, 'streamlit app boots', str(at.exception)[:200])
                return
            at.session_state['toto_game'] = 'german'
            at.session_state['coupon_german'] = (
                'Norway - Italy\nTurkey - Spain\nBrazil - Ghana')
            at.run()
            btn = [b for b in at.button if 'Analyze' in b.label]
            if not btn:
                record(FAIL, 'streamlit app boots', 'Analyze button missing')
                return
            btn[0].click(); at.run()
            if at.exception:
                record(FAIL, 'toto tab analyses a coupon',
                       str(at.exception)[:200])
                return
            shown = [d.value for d in at.dataframe
                     if 'Pick' in list(getattr(d.value, 'columns', []))]
            record(PASS if shown else FAIL, 'streamlit app + toto tab',
                   f'{len(shown[-1])} matches analysed' if shown
                   else 'no result table')
        finally:
            os.environ.pop('FOOTBALL_PREDICTOR_TOTO_DIR', None)


def main() -> int:
    ap = argparse.ArgumentParser(description='Predictor health check')
    ap.add_argument('--quick', action='store_true',
                    help='skip the slow app boot and check fewer leagues')
    args = ap.parse_args()

    print('=' * 64)
    print('  FOOTBALL PREDICTOR — SELF TEST')
    print('=' * 64)

    check_config_wiring()
    check_league_files()
    df = check_processed_leagues()
    check_freshness(df)
    check_team_leakage(df)
    check_toto_math()
    check_name_resolution()
    check_tracker()
    check_models(args.quick)
    check_international()
    if not args.quick:
        check_app()

    fails = [r for r in _results if r[0] == FAIL]
    warns = [r for r in _results if r[0] == WARN]
    print('=' * 64)
    print(f"  {len(_results) - len(fails) - len(warns)} passed, "
          f"{len(warns)} warning(s), {len(fails)} failed")
    print('=' * 64)
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
