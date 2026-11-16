"""
The predictor: the best model for each task, in one place
=========================================================
One call gives every number the app shows for a match:

    1X2            bookmaker odds when the match has them (they beat every
                   model on matches no model had seen); otherwise the v2
                   ensemble (XGBoost + LightGBM on causal features and market
                   ratings)
    over/under     read off the score grid; pooled with the over/under odds
                   when the match has them (weights fitted on the tuning years)
    both score     read off the score grid
    correct score  the score grid: the v2 scoreline model, mixed with the
                   grid the 1X2 odds imply, shape-corrected, then re-weighted
                   so it agrees with the final 1X2 and over/under numbers.
                   Every market is read off that one grid, so the numbers can
                   never contradict each other.

Why these choices — measured on matches the models had not seen — is written
down in V2_RESULTS.md (scripts/v2_grid_calib.py for the grid).

Matches the club model cannot serve are priced by the fallback that fits them:
national teams by the international Elo + goal model (with home advantage
unless the match is on neutral ground), clubs from different leagues by global
club Elo. A pair no model knows is still priced when it has odds: the 1X2 from
the odds, the grid from the goal rates those odds imply.
"""

from __future__ import annotations

import os
import threading

import numpy as np
import pandas as pd

from scripts import config, market, names
from scripts import research_features as rf
from scripts import v2_goals, v2_bundle, v2_grid_calib

G = v2_goals.G
SHOW = 5                                   # grid cells 0..4 are shown
LABELS = ('1', 'X', '2')
# After this many days of a season every club in the league has played. A
# club with no match in the season is then not in the league (relegated, or
# gone), and what the model knows about it is old.
SEASON_SETTLED_DAYS = 21


def _odds(values, n: int):
    """A list of n valid decimal odds, or None."""
    try:
        o = [float(x) for x in values] if values is not None and len(values) == n else None
    except (TypeError, ValueError):
        return None
    return o if o and all(np.isfinite(x) and x > 1 for x in o) else None


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def pool_logit(p_model, p_book, w_model: float, w_book: float) -> float:
    z = w_model * _logit(p_model) + w_book * _logit(p_book)
    return float(1 / (1 + np.exp(-z)))


class Predictor:
    def __init__(self):
        self.lock = threading.RLock()
        self.reload()

    # ------------------------------------------------------------------ data
    def reload(self) -> None:
        """(Re)read the match files. Cheap enough to do after every update."""
        raw, _ = rf.load_raw()
        cols = ['league', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'SeasonId']
        self.results = raw[cols].copy()
        latest = self.results.sort_values('Date', ascending=False, kind='stable')
        clubs: dict[str, str] = {}
        for lg, h, a in zip(latest.league, latest.HomeTeam, latest.AwayTeam):
            clubs.setdefault(h, lg)
            clubs.setdefault(a, lg)
        self.clubs = clubs
        self.last_result = {lg: pd.Timestamp(d) for lg, d in
                            self.results.groupby('league')['Date'].max().items()}
        # who plays in each league this season, and each club's newest match there
        long = pd.concat([self.results[['league', 'Date', 'HomeTeam']].rename(columns={'HomeTeam': 'team'}),
                          self.results[['league', 'Date', 'AwayTeam']].rename(columns={'AwayTeam': 'team'})])
        self.team_last = long.groupby(['league', 'team'])['Date'].max().to_dict()
        self.current = {}
        for lg, d in self.results.groupby('league'):
            season = d.loc[d.Date.idxmax(), 'SeasonId']
            now = d[d.SeasonId == season]
            self.current[lg] = {'start': now.Date.min(), 'last': d.Date.max(),
                                'teams': set(now.HomeTeam) | set(now.AwayTeam)}
        from scripts import toto
        toto._INTL_CACHE = None
        self.intl = toto._load_intl()
        self.index = names.NameIndex(clubs, sorted(self.intl['ratings']))
        self._nation = {names.norm(n): n for n in self.intl['ratings']}
        self._bundle = None

    @property
    def data_through(self) -> pd.Timestamp:
        return max(self.last_result.values())

    def bundles(self):
        """(1X2 model, goals model, pool weights) — loaded once."""
        if self._bundle is None:
            m1, meta1 = v2_bundle.serving_bundle()
            mg, metag = v2_bundle.serving_goals()
            w = metag.get('ou25_with_odds') or {'model': 0.118, 'book': 1.106}
            self._bundle = (m1, mg, w, meta1, metag)
            self.calib = v2_grid_calib.load_calibration()      # None: plain tilt
        return self._bundle

    def info(self) -> dict:
        _, _, _, m1, mg = self.bundles()
        return {'ok': True, '1x2': m1['tag'], 'goals': mg['tag'],
                'trained_through': m1['trained_through']}

    # ----------------------------------------------------------------- names
    def side(self, typed: str) -> dict:
        """One typed team name -> canonical name, display name, kind, how it
        matched, and suggestions when it did not."""
        m = self.index.resolve(typed)
        if m:
            return {'canonical': m.canonical, 'display': m.display, 'kind': m.kind,
                    'how': m.how, 'league': m.league, 'suggest': []}
        try:                                  # a club we have no league data for
            from scripts import clubs_europe
            if clubs_europe.resolve(typed):
                return {'canonical': typed, 'display': typed, 'kind': 'euro',
                        'how': 'euro', 'league': None, 'suggest': []}
        except Exception:
            pass
        sug = [{'name': c, 'display': self.index.display_of[c],
                'league_name': config.LEAGUE_REGISTRY.get(self.index.entries[c][1], {}).get(
                    'display_name', '') if self.index.entries[c][0] == 'club' else 'National team'}
               for c, _ in self.index.suggest(typed, 5)]
        return {'canonical': None, 'display': typed, 'kind': None, 'how': None,
                'league': None, 'suggest': sug}

    # --------------------------------------------------------------- predict
    def _sides(self, home: str, away: str):
        """Both names resolved. A name that is a club AND a national team
        (Andorra, Monaco) follows its opponent: against a national team it is
        the national team."""
        sh, sa = self.side(home), self.side(away)
        for mine, other, typed in ((sh, sa, home), (sa, sh, away)):
            nation = self._nation.get(names.norm(typed))
            if nation and other['kind'] == 'national' and mine['kind'] != 'national':
                mine.update(canonical=nation, display=nation, kind='national', how='nation',
                            league=None, suggest=[])
        return sh, sa

    def predict(self, home: str, away: str, odds=None, ou_odds=None, date=None,
                neutral: bool = False) -> dict:
        """Everything for one match. `odds` = decimal [1, X, 2]; `ou_odds` =
        decimal [over 2.5, under 2.5]; `neutral` = a national-team match on
        neutral ground. Never raises for an unknown team: without odds the
        result has ok=False and the suggestions; with odds it is priced from
        the odds alone."""
        with self.lock:
            sh, sa = self._sides(home, away)
            out = {'ok': False, 'home': home, 'away': away, 'sides': {'home': sh, 'away': sa},
                   'home_display': sh['display'], 'away_display': sa['display'],
                   'kind': None, 'league': None, 'league_name': '', 'src': 'none'}
            o, u = _odds(odds, 3), _odds(ou_odds, 2)
            core = None
            if sh['canonical'] and sa['canonical']:
                ch, ca = sh['canonical'], sa['canonical']
                out['home'], out['away'] = ch, ca
                if sh['kind'] == 'club' and sa['kind'] == 'club' and sh['league'] == sa['league']:
                    core = self._club(ch, ca, sh['league'], date)
                if core is None and sh['kind'] == 'national' and sa['kind'] == 'national':
                    core = self._nations(ch, ca, neutral)
                if core is None and sh['kind'] in ('club', 'euro') and sa['kind'] in ('club', 'euro'):
                    core = self._euro(ch, ca)
            if core is None and o:
                core = {'kind': 'odds', 'league': None, 'grid': None, 'p1x2': None}
            if core is None:
                return out
            return self._finish(out, core, o, u)

    # ---- model paths: each returns {'grid', 'p1x2' (model, [1,X,2]), 'kind', ...}
    def _old_data(self, league: str, team: str):
        """The day of the team's newest match in this league when it is not
        one of the league's clubs this season; None for a current club."""
        cur = self.current.get(league)
        if not cur or team in cur['teams']:
            return None
        if (cur['last'] - cur['start']).days < SEASON_SETTLED_DAYS:
            return None                       # too early to say: it may not have played yet
        return self.team_last.get((league, team))

    def _club(self, home, away, league, date):
        m1, mg, _, meta1, metag = self.bundles()
        flags = v2_bundle._serving_flags(meta1)
        state = v2_bundle.serving_state(league, **flags)
        if (league, home) not in state.teams or (league, away) not in state.teams:
            return None
        day = pd.Timestamp(date if date is not None else pd.Timestamp.now()).normalize()
        day = max(day, state.last_day + pd.Timedelta(days=1))
        row = {'Date': day, 'league': league, 'HomeTeam': home, 'AwayTeam': away}
        feat = pd.DataFrame([dict(state.fixture(row), **row)])
        pa, pd_, ph = m1.predict_proba(feat)[0]                 # model order [A, D, H]
        grid = mg.predict_grid(feat)[0]
        old = {side: self._old_data(league, team) for side, team in (('home', home), ('away', away))}
        return {'kind': 'club', 'league': league, 'grid': grid, 'p1x2': [ph, pd_, pa],
                'old': {k: v for k, v in old.items() if v is not None}}

    def _nations(self, home, away, neutral=False):
        ic = self.intl
        if ic['model'] is None or home not in ic['ratings'] or away not in ic['ratings']:
            return None
        d = ic['model'].market_probs(ic['ratings'][home], ic['ratings'][away], neutral=bool(neutral))
        return {'kind': 'national', 'league': None, 'grid': np.asarray(d['grid']),
                'p1x2': [d['p_home'], d['p_draw'], d['p_away']], 'neutral': bool(neutral)}

    def _euro(self, home, away):
        try:
            from scripts import clubs_europe
            r = clubs_europe.current_ratings()
            eh, ea = self._elo(home, r), self._elo(away, r)
            mdl = clubs_europe.fitted_model()
            if r is None or eh is None or ea is None or mdl is None:
                return None
            d = mdl.market_probs(eh, ea, neutral=False)
        except Exception:
            return None
        grid = v2_goals.grid(np.array([d['lambda_home']]), np.array([d['lambda_away']]), 0.0)[0]
        return {'kind': 'euro', 'league': None, 'grid': grid,
                'p1x2': [d['p_home'], d['p_draw'], d['p_away']]}

    def _elo(self, name, ratings):
        """Global club rating under any name the club goes by: the data name
        ('Buyuksehyr'), the full name ('İstanbul Başakşehir'), nicknames."""
        from scripts import clubs_europe
        tried = [name]
        if name in self.index.entries:
            tried += [self.index.display_of[name]] + self.index.aliases_of[name]
        for n in tried:
            e = clubs_europe.elo_of(n, ratings)
            if e is not None:
                return e
        return None

    # ------------------------------------------------------- the common part
    def _finish(self, out, core, o, u) -> dict:
        kind = core['kind']
        self.bundles()
        w = self._bundle[2] if kind == 'club' else {'model': .118, 'book': 1.106}
        has_model = core.get('grid') is not None
        grid = core['grid'][None, :, :] if has_model else None
        model_p = None
        if has_model:
            model_p = np.asarray(core['p1x2'], float)
            model_p = model_p / model_p.sum()
        mk = np.asarray(market.shin(o), float) if o else None          # [1, X, 2]
        final = mk if mk is not None else model_p
        raw_ou = float(v2_goals.grid_markets(grid)['ou25'][0]) if has_model else None
        ou_target, ou_mkt = None, None
        if u:
            ou_mkt = (1 / u[0]) / (1 / u[0] + 1 / u[1])
            ou_target = pool_logit(raw_ou, ou_mkt, w['model'], w['book']) if kind == 'club' else ou_mkt
        # the joint distribution everything is read from. The calibration was
        # measured on club matches and on odds-only pairs; the national and
        # club-Elo grids keep the plain tilt.
        calib = self.calib if kind in ('club', 'odds') else None
        rates = None
        if mk is not None and (calib or not has_model):
            rates = v2_goals.rates_from_1x2(mk[None, :], float((calib or {}).get('rho', v2_goals.ODDS_RHO)))
        tilted = v2_goals.serve(grid, final[None, :],
                                ou25=np.array([ou_target]) if ou_target is not None else None,
                                odds_rates=rates, calib=calib)
        m = v2_goals.grid_markets(tilted)
        # how exactly the grid ended up on the numbers it was asked to match
        # (zero unless the 1X2 and over/under numbers contradict each other)
        miss = np.abs(np.array([m['pH'][0], m['pD'][0], m['pA'][0]]) - final).max()
        if ou_target is not None:
            miss = max(miss, abs(float(m['ou25'][0]) - ou_target))
        g = tilted[0]
        flat = g.ravel()
        order = np.argsort(-flat)[:6]
        scores = [{'score': f'{k // (G + 1)}-{k % (G + 1)}', 'p': float(flat[k])} for k in order]
        i, j = np.indices(g.shape)
        league = core.get('league')
        league_name = config.LEAGUE_REGISTRY.get(league, {}).get('display_name', '') if league else ''
        old = {side: str(pd.Timestamp(day).date()) for side, day in (core.get('old') or {}).items()}
        notes = []
        if kind == 'odds':
            notes.append('No model knows this pair. Every number comes from the odds.')
        for side, day in (core.get('old') or {}).items():
            notes.append(f"{out[side + '_display']} is not in the {league_name} data this season. "
                         f"Its newest match there is from {pd.Timestamp(day):%B %Y}, "
                         f"so the model numbers rest on old data.")
        if kind == 'national':
            notes.append('Priced as a match on neutral ground.' if core.get('neutral') else
                         f"Home advantage for {out['home_display']} is included.")
        out.update({
            'ok': True, 'kind': kind, 'league': league, 'league_name': league_name,
            'p1x2': [float(x) for x in final],
            'src': 'odds' if mk is not None else ('model' if kind == 'club' else kind),
            'model_1x2': [float(x) for x in model_p] if model_p is not None else None,
            'market_1x2': [float(x) for x in mk] if mk is not None else None,
            'odds': [float(x) for x in o] if mk is not None else None,
            'ou': {'1.5': float(m['ou15'][0]), '2.5': float(m['ou25'][0]), '3.5': float(m['ou35'][0])},
            'ou_src': 'odds' if ou_target is not None else ('model' if has_model else 'implied'),
            'ou25_model': raw_ou, 'ou25_market': ou_mkt, 'fit_error': float(miss),
            'btts': float(m['btts'][0]),
            'xg': [float(m['xg_h'][0]), float(m['xg_a'][0])],
            'scores': scores,
            'grid': [[float(g[a, b]) for b in range(SHOW)] for a in range(SHOW)],
            'grid_rest': float(g[(i >= SHOW) | (j >= SHOW)].sum()),
            'old_data': old or None, 'notes': notes,
            'neutral': core.get('neutral') if kind == 'national' else None,
        })
        if out['league']:
            last = self.last_result.get(out['league'])
            out['data_age_days'] = int((pd.Timestamp.today().normalize() - last).days) if last is not None else None
        return out
