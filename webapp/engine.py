"""
Engine behind the web app
=========================
Loads the heavy context once (match data, team tables, national-team model)
and exposes plain functions — JSON-ready dicts in, dicts out — over the
existing modules. Nothing here re-implements a model or the Toto maths:
probabilities come from scripts/toto.match_probs (odds > club model v2/v1 >
European clubs > national teams), tickets from scripts/ticket.build, history
from scripts/track, names from scripts/names.

One coupon, no game switch. Both pools pay a prize for at most three wrong
(13er Wette: 10+ of 13, Spor Toto: 12+ of 15), so the coupon is simply "the
matches you added" and its size decides what it is: 13 plays as the 13er
Wette, 15 as Spor Toto, anything else is previewed with the same rule.

The coupon lives in data/_toto/coupon.txt in the one-line-per-match format
the classic app uses ("Home - Away  1.95 3.40 3.90"); what that format cannot
hold (crowd %, locked symbols, budget) lives next to it in coupon.meta.json.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts import config, names, toto  # noqa: E402
from scripts import predict as predict_mod  # noqa: E402
from scripts import ticket as ticket_mod  # noqa: E402

COUPON_DIR = os.environ.get('FOOTBALL_PREDICTOR_TOTO_DIR',
                            os.path.join(config.DATA_DIR, '_toto'))
SETTINGS_FILE = os.path.join(COUPON_DIR, 'settings.json')
COUPON_FILE = os.path.join(COUPON_DIR, 'coupon.txt')
COUPON_META = os.path.join(COUPON_DIR, 'coupon.meta.json')

ALLOWED_WRONG = 3                      # both pools: a prize needs at most 3 wrong
GAMES = [  # (matches, label, flag, prize_from)
    (13, '13er Wette', '🇩🇪', 10),
    (15, 'Spor Toto', '🇹🇷', 12),
]
FIXTURE_DAYS = 14
FIXTURE_TTL = 15 * 60
# Model calls touch shared caches (v2 per-league states, the intl model) that
# are not safe to fill from two threads at once; one user, so serialise.
MODEL_LOCK = threading.RLock()


def game_for(n: int) -> dict:
    """What a coupon of n matches is."""
    for size, label, flag, prize in GAMES:
        if n == size:
            return {'n': n, 'label': label, 'flag': flag, 'prize_from': prize,
                    'complete': True, 'key': 'german' if size == 13 else 'turkish'}
    return {'n': n, 'label': None, 'flag': '', 'complete': False,
            'prize_from': max(1, n - ALLOWED_WRONG),
            'key': 'german' if n <= 13 else 'turkish'}


def _disp(league: str | None) -> str:
    if not league:
        return ''
    return config.LEAGUE_REGISTRY.get(league, {}).get('display_name', league)


def _hda(d: dict | None):
    """{'home','draw','away'} -> [1, X, 2] floats, or None."""
    if not d:
        return None
    return [float(d['home']), float(d['draw']), float(d['away'])]


def _atomic_write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(text)
    os.replace(tmp, path)


def row_key(home: str, away: str) -> str:
    """Order-insensitive, accent-folded identity of a fixture."""
    return '|'.join(sorted((names.norm(home), names.norm(away))))


class Engine:
    def __init__(self):
        self.state, self.step, self.error = 'starting', '', None
        self.ctx = None
        self.index: names.NameIndex | None = None
        self.teams: list[dict] = []
        self.loaded_at = None
        self.fx = {'state': 'idle', 'done': 0, 'total': 0, 'rows': [], 'mode': None,
                   'loaded_at': None, 'error': None}
        self._fx_lock = threading.Lock()
        self._job = {'running': False, 'kind': None, 'log': '', 'ok': None,
                     'started': None, 'finished': None}

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        """Heavy loading; run in a background thread so the page can show
        progress instead of hanging."""
        try:
            self.state = 'loading'
            self.step = 'Reading match data'
            from scripts import data_loader, utils
            hist = data_loader.load_processed_data()
            self.step = 'Building team tables'
            team_stats = utils.get_team_stats_table(hist)
            t2l = utils.get_team_to_league_map(hist)
            self.step = 'National-team model'
            ic = toto._load_intl()
            self.step = 'Team names'
            predict_mod.set_model_mode(self.settings().get('model_mode', 'v2'))
            self.index = names.NameIndex(t2l, sorted(ic['ratings']))
            self.teams = self._team_list()
            last = hist.groupby('league')['Date'].max()
            self.ctx = {'hist': hist, 'team_stats': team_stats,
                        'team_to_league': t2l, 'teams': set(t2l),
                        'weights': toto._load_blend_weights(),
                        'last_result': {lg: pd.Timestamp(d) for lg, d in last.items()}}
            self.fx = {'state': 'idle', 'done': 0, 'total': 0, 'rows': [], 'mode': None,
                       'loaded_at': None, 'error': None}
            self.loaded_at = time.time()
            self.state, self.step = 'ready', ''
            self.start_fixtures()                 # prices the week ahead in the background
        except Exception as e:                           # shown on the page
            self.state, self.error = 'error', f'{type(e).__name__}: {e}'
            traceback.print_exc()

    def start(self) -> None:
        threading.Thread(target=self.load, daemon=True).start()

    def need_ready(self):
        if self.state != 'ready':
            raise RuntimeError('Still warming up — try again in a few seconds.')

    def _team_list(self) -> list[dict]:
        idx = self.index
        out = []
        for canon, (kind, lg) in idx.entries.items():
            out.append({'name': canon, 'display': idx.display_of[canon], 'kind': kind,
                        'league': lg, 'league_name': _disp(lg) if kind == 'club' else 'National team',
                        'keys': idx.search_keys(canon)})
        out.sort(key=lambda t: (t['kind'] != 'club', names.norm(t['display'])))
        return out

    # -------------------------------------------------------------- settings
    def settings(self) -> dict:
        try:
            with open(SETTINGS_FILE, encoding='utf-8') as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def set_model_mode(self, mode: str) -> dict:
        predict_mod.set_model_mode(mode)                 # validates
        s = self.settings()
        s['model_mode'] = mode
        _atomic_write(SETTINGS_FILE, json.dumps(s, indent=2))
        self.start_fixtures(force=True)
        return self.status()

    # ---------------------------------------------------------------- status
    def status(self) -> dict:
        out = {'state': self.state, 'step': self.step, 'error': self.error,
               'model_mode': predict_mod.get_model_mode(),
               'allowed_wrong': ALLOWED_WRONG,
               'games': [{'n': n, 'label': lab, 'flag': flag, 'prize_from': pf}
                         for n, lab, flag, pf in GAMES],
               'job': {k: v for k, v in self._job.items() if k != 'log'}}
        try:
            from scripts import v2_bundle
            info = v2_bundle.serving_info()
            out['v2'] = info
            if os.path.isfile(v2_bundle.LATEST_GOALS):
                with open(v2_bundle.LATEST_GOALS) as f:
                    out['v2']['goals_tag'] = f.read().strip()
        except Exception as e:
            out['v2'] = {'ok': False, 'error': str(e)}
        if self.ctx:
            today = pd.Timestamp.today().normalize()
            last = self.ctx['last_result']
            out['data_through'] = str(max(last.values()).date())
            stale = sorted(((int((today - d).days), lg) for lg, d in last.items()
                            if (today - d).days > 28), reverse=True)
            out['stale'] = [{'league': lg, 'name': _disp(lg), 'days': d} for d, lg in stale]
            out['n_leagues'] = len(last)
            out['n_teams'] = len(self.ctx['team_to_league'])
        try:
            from scripts import international as intl
            out['intl_updated_h'] = round((time.time() - os.path.getmtime(intl.INTL_FILE)) / 3600, 1)
        except OSError:
            pass
        return out

    # ---------------------------------------------------------------- coupon
    def load_coupon(self) -> dict:
        try:
            with open(COUPON_FILE, encoding='utf-8') as f:
                text = f.read()
        except OSError:
            text = ''
        try:
            with open(COUPON_META, encoding='utf-8') as f:
                meta = json.load(f)
        except (OSError, ValueError):
            meta = {}
        per = meta.get('rows', {})
        rows = []
        for _, r in toto.parse_lines(text).iterrows():
            odds = [float(r[c]) for c in ('o1', 'ox', 'o2')] \
                if all(pd.notna(r.get(c)) for c in ('o1', 'ox', 'o2')) else None
            extra = per.get(row_key(r['home'], r['away']), {})
            rows.append({'home': str(r['home']), 'away': str(r['away']), 'odds': odds,
                         'crowd': extra.get('crowd'), 'lock': extra.get('lock')})
        return {'rows': rows, 'budget': int(meta.get('budget', 1)),
                'tilt': float(meta.get('tilt', ticket_mod.DEFAULT_TILT)),
                'objective': meta.get('objective', 'hit'),
                'updated': meta.get('updated')}

    def save_coupon(self, body: dict) -> dict:
        lines, per = [], {}
        for r in body.get('rows', []):
            home, away = str(r.get('home') or '').strip(), str(r.get('away') or '').strip()
            if not home and not away:
                continue
            line = f"{home} - {away}"
            odds = r.get('odds')
            if odds and len(odds) == 3 and all(o and float(o) > 1 for o in odds):
                line += '  ' + ' '.join(f'{float(o):.2f}' for o in odds)
            lines.append(line)
            extra = {}
            if r.get('crowd') and len(r['crowd']) == 3:
                extra['crowd'] = [float(x) for x in r['crowd']]
            if r.get('lock') in ticket_mod.MASK_LABEL.values():
                extra['lock'] = r['lock']
            if extra:
                per[row_key(home, away)] = extra
        _atomic_write(COUPON_FILE, '\n'.join(lines))
        meta = {'rows': per, 'budget': int(body.get('budget', 1)),
                'tilt': float(body.get('tilt', ticket_mod.DEFAULT_TILT)),
                'objective': body.get('objective', 'hit'),
                'updated': pd.Timestamp.now().isoformat(timespec='seconds')}
        _atomic_write(COUPON_META, json.dumps(meta, indent=2, ensure_ascii=False))
        return {'ok': True, 'n': len(lines), 'updated': meta['updated']}

    def legacy(self) -> dict:
        """Rows of the two coupons the classic app kept, to import once."""
        out = {}
        for game in ('turkish', 'german'):
            try:
                with open(os.path.join(COUPON_DIR, f'{game}.txt'), encoding='utf-8') as f:
                    rows = self.parse_text(f.read())
            except OSError:
                rows = []
            if rows:
                out[game] = rows
        return out

    def parse_text(self, text: str) -> list[dict]:
        out = []
        for _, r in toto.parse_lines(text).iterrows():
            odds = [float(r[c]) for c in ('o1', 'ox', 'o2')] \
                if all(pd.notna(r.get(c)) for c in ('o1', 'ox', 'o2')) else None
            out.append({'home': str(r['home']), 'away': str(r['away']), 'odds': odds})
        return out

    # ----------------------------------------------------------------- names
    def remember_name(self, typed: str, canonical: str) -> dict:
        self.need_ready()
        return {'ok': self.index.remember(typed, canonical)}

    def _side(self, typed: str) -> dict:
        """One team name -> {canonical, display, kind, how, suggest}."""
        m = self.index.resolve(typed)
        if m:
            return {'canonical': m.canonical, 'display': m.display, 'kind': m.kind,
                    'how': m.how, 'league': m.league, 'suggest': []}
        # a club outside the modelled leagues can still be priced via club Elo
        try:
            from scripts import clubs_europe
            if clubs_europe.resolve(typed):
                return {'canonical': typed, 'display': typed, 'kind': 'euro',
                        'how': 'euro', 'league': None, 'suggest': []}
        except Exception:
            pass
        sug = [{'name': c, 'display': self.index.display_of[c],
                'league_name': _disp(self.index.entries[c][1])
                if self.index.entries[c][0] == 'club' else 'National team'}
               for c, _ in self.index.suggest(typed, 5)]
        return {'canonical': None, 'display': typed, 'kind': None, 'how': None,
                'league': None, 'suggest': sug}

    # ------------------------------------------------------------- analysis
    def analyze_row(self, home: str, away: str, odds=None) -> dict:
        """Probabilities for one coupon row, where they came from, both club
        model versions, and name suggestions for anything unrecognised."""
        self.need_ready()
        home, away = str(home or '').strip(), str(away or '').strip()
        out = {'home': home, 'away': away, 'probs': None, 'src': 'none',
               'v1': None, 'v2': None, 'market': None, 'league': None, 'league_name': '',
               'sides': {'home': None, 'away': None}}
        if not home or not away:
            return out
        sh, sa = self._side(home), self._side(away)
        out['sides'] = {'home': sh, 'away': sa}
        ch, ca = sh['canonical'] or home, sa['canonical'] or away
        o = list(odds) if odds and len(odds) == 3 and all(x and float(x) > 1 for x in odds) else None
        row = pd.Series({'home': ch, 'away': ca,
                         'o1': o[0] if o else None, 'ox': o[1] if o else None,
                         'o2': o[2] if o else None})
        ctx = dict(self.ctx)
        ctx['model_detail'] = {}
        with MODEL_LOCK:
            p, src, model = toto.match_probs(row, ctx)
        det = ctx['model_detail'].get((ch, ca))
        if det:
            out['v1'], out['v2'] = _hda(det['v1']), _hda(det['v2'])
            out['league'], out['league_name'] = det['league'], _disp(det['league'])
        if o:
            out['market'] = [float(x) for x in toto._devig(*o)]
        if p is not None:
            out['probs'] = [float(x) for x in p]
            if src in ('blend', 'odds'):
                out['src'] = 'odds'
            elif src == 'model' and det:
                out['src'] = 'odds' if det['feed_odds'] else det['version']
            else:
                out['src'] = src                         # euro / intl
        return out

    # --------------------------------------------------------------- tickets
    @staticmethod
    def _tiers(probs, masks, threshold, top):
        qs = [float(sum(p[o] for o in range(3) if (m >> o) & 1)) for p, m in zip(probs, masks)]
        d = toto.pb_distribution(qs)
        return ({str(k): float(d[k:].sum()) if k < len(d) else 0.0
                 for k in range(threshold, top + 1)}, float(sum(qs)))

    @staticmethod
    def threshold_for(n: int) -> int:
        return max(1, n - ALLOWED_WRONG)

    def ticket(self, body: dict) -> dict:
        probs = [np.asarray(p, float) for p in body['probs']]
        if not probs:
            return {'labels': [], 'masks': [], 'columns': 0}
        n = len(probs)
        thr = self.threshold_for(n)
        budget = max(1, int(body.get('budget', 1)))
        inv = {v: k for k, v in ticket_mod.MASK_LABEL.items()}
        fixed = {i: inv[l] for i, l in enumerate(body.get('locks') or []) if l in inv}
        crowd = body.get('crowd')
        use_crowd = (body.get('objective') == 'crowd' and crowd
                     and len(crowd) == n and all(c and len(c) == 3 for c in crowd))
        t0 = time.time()
        r = ticket_mod.build(probs, thr, budget,
                             crowd=np.asarray(crowd, float) if use_crowd else None,
                             tilt=float(body.get('tilt', ticket_mod.DEFAULT_TILT)),
                             fixed=fixed, n_sims=12000, restarts=4)
        tiers, exp = self._tiers(probs, r['masks'], thr, n)
        single = [1 << int(np.argmax(p)) for p in probs]
        s_tiers, s_exp = self._tiers(probs, single, thr, n)
        return {'labels': r['labels'], 'masks': r['masks'], 'columns': r['columns'],
                'budget': budget, 'objective': r['objective'],
                'over_budget': r.get('over_budget', False),
                'threshold': thr, 'n': n, 'game': game_for(n), 'tiers': tiers,
                'p_prize': tiers[str(thr)], 'expected': exp,
                'single': {'p_prize': s_tiers[str(thr)], 'expected': s_exp},
                'n_double': sum(1 for m in r['masks'] if ticket_mod.popcount(m) == 2),
                'n_triple': sum(1 for m in r['masks'] if ticket_mod.popcount(m) == 3),
                'ms': int((time.time() - t0) * 1000)}

    def ladder(self, body: dict) -> list[dict]:
        """What each extra doubling of the budget buys (hit objective)."""
        probs = [np.sort(np.asarray(p, float))[::-1] for p in body['probs']]
        thr = self.threshold_for(len(probs))
        out, prev = [], None
        for b in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512):
            cov, cols, p = toto.optimize_system(probs, thr, b, restarts=3)
            if prev is not None and cols == prev:
                continue
            prev = cols
            out.append({'budget': b, 'columns': cols, 'p_prize': float(p),
                        'doubles': sum(1 for c in cov if c == 2),
                        'triples': sum(1 for c in cov if c == 3)})
        return out

    def sweep(self, body: dict) -> list[dict]:
        probs = [np.asarray(p, float) for p in body['probs']]
        thr = self.threshold_for(len(probs))
        rows = ticket_mod.sweep(probs, thr, max(1, int(body.get('budget', 1))),
                                np.asarray(body['crowd'], float), n_sims=20000)
        return [{'tilt': r['tilt'], 'p_hit': r['p_hit'], 'payout': r['vs_base'],
                 'ticket': r['ticket'], 'columns': r['columns']} for r in rows]

    # ------------------------------------------------------------------ odds
    def fill_odds(self, rows: list[dict]) -> list:
        from scripts import fixtures as fx_mod
        fx_mod.fetch()
        feed = fx_mod.load(fetch_if_missing=False)
        out = []
        for r in rows:
            if r.get('odds') or not r.get('home') or not r.get('away'):
                out.append(None)
                continue
            o = fx_mod.find_odds_any_league(feed, r['home'], r['away'])
            out.append([o['OddsH'], o['OddsD'], o['OddsA']] if o else None)
        return out

    # -------------------------------------------------------------- fixtures
    def start_fixtures(self, force: bool = False, refetch: bool = False) -> None:
        """Price the next two weeks of the odds feed in the background. The
        list is available immediately (teams, odds) and fills in as each match
        is priced, so the page never waits on the models."""
        if self.state != 'ready':
            return
        mode = predict_mod.get_model_mode()
        with self._fx_lock:
            fx = self.fx
            if fx['state'] == 'loading':
                return
            fresh = (fx['state'] == 'ready' and fx['mode'] == mode
                     and time.time() - (fx['loaded_at'] or 0) < FIXTURE_TTL)
            if fresh and not force and not refetch:
                return
            self.fx = {'state': 'loading', 'done': 0, 'total': 0, 'rows': [], 'mode': mode,
                       'loaded_at': None, 'error': None, 'started': time.time()}
        threading.Thread(target=self._price_fixtures, args=(refetch,), daemon=True).start()

    def _price_fixtures(self, refetch: bool) -> None:
        fx = self.fx
        try:
            from scripts import fixtures as fx_mod
            fx_mod.fetch(force=refetch)
            feed = fx_mod.load(fetch_if_missing=False)
            now = pd.Timestamp.now().normalize()
            feed = feed[(feed['Date'] >= now) & (feed['Date'] <= now + pd.Timedelta(days=FIXTURE_DAYS))]
            feed = feed.dropna(subset=['league', 'HomeTeam', 'AwayTeam', 'Date'])
            feed = feed.drop_duplicates(['league', 'Date', 'HomeTeam', 'AwayTeam'])
            feed = feed.sort_values(['Date', 'Time', 'league', 'HomeTeam'], na_position='last')
            idx = self.index
            base = []
            for k, m in enumerate(feed.to_dict('records')):
                odds = [m['OddsH'], m['OddsD'], m['OddsA']]
                odds = [float(x) for x in odds] if all(pd.notna(x) and x > 1 for x in odds) else None
                market = [float(x) for x in toto._devig(*odds)] if odds else None
                home, away, lg = m['HomeTeam'], m['AwayTeam'], m['league']
                base.append({
                    'id': f'{lg}|{m["Date"].date()}|{home}|{away}', 'date': str(m['Date'].date()),
                    'time': (str(m['Time']) if pd.notna(m.get('Time')) else None),
                    'league': lg, 'league_name': _disp(lg),
                    'home': home, 'away': away,
                    'home_display': idx.display_of.get(home, home), 'away_display': idx.display_of.get(away, away),
                    'odds': odds, 'market': market, 'probs': market, 'src': 'odds' if market else None,
                    'v1': None, 'v2': None, 'ou25': None, 'btts': None, 'goals_v1': None, 'goals_v2': None,
                    'pending': True})
            fx['rows'] = base
            fx['total'] = len(base)
            try:                                       # build league states in parallel first
                from scripts import v2_bundle
                v2_bundle.warm(sorted({r['league'] for r in base}))
            except Exception:
                traceback.print_exc()
            c = self.ctx
            for i, r in enumerate(base):
                new = dict(r, pending=False)
                try:
                    with MODEL_LOCK:
                        pred = predict_mod.predict_match(
                            r['home'], r['away'], c['team_stats'], c['team_to_league'], c['hist'],
                            include_xg=False, prediction_date=pd.Timestamp(r['date']))
                    new['v1'], new['v2'] = _hda(pred.get('1x2_v1')), _hda(pred.get('1x2_v2'))
                    g1, g2 = pred.get('goals_v1') or {}, pred.get('goals_v2') or {}
                    new['goals_v1'] = {'ou25': (g1.get('ou25') or {}).get('over'), 'btts': (g1.get('btts') or {}).get('yes')}
                    new['goals_v2'] = {'ou25': (g2.get('ou25') or {}).get('over'), 'btts': (g2.get('btts') or {}).get('yes')} if g2 else None
                    if 'ou25' in pred:
                        new['ou25'] = pred['ou25']['over']
                    if 'btts' in pred:
                        new['btts'] = pred['btts']['yes']
                    new['model_version'] = pred.get('model_version')
                    if not new['market']:
                        used = _hda(pred.get('1x2'))
                        new['probs'], new['src'] = used, (pred.get('model_version') or 'v1')
                except Exception:
                    pass                                # leaves the odds-only row
                fx['rows'][i] = new
                fx['done'] = i + 1
            fx['loaded_at'] = time.time()
            fx['state'] = 'ready'
        except Exception as e:
            fx['state'], fx['error'] = 'error', f'{type(e).__name__}: {e}'
            traceback.print_exc()

    def fixtures(self, refresh: bool = False) -> dict:
        self.need_ready()
        self.start_fixtures(refetch=refresh)
        fx = self.fx
        return {'state': fx['state'], 'done': fx['done'], 'total': fx['total'],
                'rows': list(fx['rows']), 'mode': fx['mode'], 'error': fx.get('error'),
                'loaded_at': fx['loaded_at']}

    # ----------------------------------------------------------------- match
    def match(self, home: str, away: str) -> dict:
        """Full card for any pair: club model (both versions + goals) when the
        clubs share a league, else the European / national fallback."""
        self.need_ready()
        sh, sa = self._side(home), self._side(away)
        h, a = sh['canonical'], sa['canonical']
        t2l = self.ctx['team_to_league']
        if h and a and sh['kind'] == 'club' and sa['kind'] == 'club' and t2l.get(h) == t2l.get(a):
            with MODEL_LOCK:
                p = predict_mod.predict_match(h, a, self.ctx['team_stats'], t2l,
                                              self.ctx['hist'], include_xg=True,
                                              prediction_date=pd.Timestamp.now())
            lg = p.get('league')
            last = self.ctx['last_result'].get(lg)
            age = int((pd.Timestamp.today().normalize() - last).days) if last is not None else None
            mk = p.get('market')

            def goals(g):
                if not g:
                    return None
                return {'ou15': g.get('ou15', {}).get('over'), 'ou25': g.get('ou25', {}).get('over'),
                        'ou35': g.get('ou35', {}).get('over'), 'btts': g.get('btts', {}).get('yes'),
                        'xg': [g['xg']['home'], g['xg']['away']] if g.get('xg') else None}
            return {'kind': 'club', 'home': h, 'away': a, 'home_display': sh['display'],
                    'away_display': sa['display'], 'league': lg, 'league_name': _disp(lg),
                    'data_age_days': age,
                    'used': _hda(p.get('1x2')), 'model_version': p.get('model_version'),
                    'goals_version': p.get('goals_version'),
                    'v1': _hda(p.get('1x2_v1')), 'v2': _hda(p.get('1x2_v2')),
                    'market': _hda(mk['implied']) if mk else None,
                    'odds': [mk['odds']['home'], mk['odds']['draw'], mk['odds']['away']] if mk else None,
                    'goals_used': goals({k: p.get(k) for k in ('ou15', 'ou25', 'ou35', 'btts', 'xg')}),
                    'goals_v1': goals(p.get('goals_v1')), 'goals_v2': goals(p.get('goals_v2')),
                    'ou25_odds': (mk or {}).get('ou25_odds')}
        r = self.analyze_row(home, away)
        return {'kind': r['src'], 'home': home, 'away': away,
                'home_display': sh['display'], 'away_display': sa['display'],
                'used': r['probs'], 'sides': r['sides'], 'league_name': r['league_name']}

    # --------------------------------------------------------------- history
    def history(self) -> dict:
        from scripts import track
        graded = track.grade_all()
        entries = {e['id']: e for e in track.load_history()}
        out = []
        for g in reversed(graded):
            e = entries.get(g['id'], {})
            out.append({**{k: g[k] for k in ('id', 'game', 'saved_at', 'threshold',
                                              'p_threshold', 'columns', 'n', 'graded',
                                              'correct', 'expected', 'complete')},
                        'note': e.get('note', ''), 'kind': e.get('kind', ''),
                        'label': game_for(g['n'])['label'] or f"{g['n']} matches",
                        'flag': game_for(g['n'])['flag'],
                        'rows': [{'home': r['home'], 'away': r['away'], 'pick': r.get('pick'),
                                  'actual': r.get('actual'), 'ok': r.get('ok'),
                                  'p': [r.get('p1'), r.get('px'), r.get('p2')]}
                                 for r in g['rows']]})
        done = [g for g in out if g['complete']]
        won = [g for g in done if g['threshold'] is not None and g['correct'] >= g['threshold']]
        return {'entries': out, 'summary': {
            'completed': len(done), 'cleared': len(won),
            'correct': sum(g['correct'] for g in done),
            'expected': sum(g['expected'] for g in done)}}

    def save_history(self, body: dict) -> dict:
        from scripts import track
        n = len(body['matches'])
        matches = [{'home': m['home'], 'away': m['away'], 'p1': m['probs'][0],
                    'px': m['probs'][1], 'p2': m['probs'][2], 'pick': m['pick'],
                    'src': m.get('src')} for m in body['matches']]
        e = track.save_coupon(game_for(n)['key'], matches, budget=int(body.get('budget', 1)),
                              threshold=self.threshold_for(n), p_threshold=float(body['p_prize']),
                              note=body.get('note', ''), columns=int(body['columns']),
                              kind=body.get('kind', 'ticket'))
        return {'ok': True, 'id': e['id']}

    def delete_history(self, entry_id: str) -> dict:
        from scripts import track
        return {'ok': track.delete_coupon(entry_id)}

    # ------------------------------------------------------------ data jobs
    def start_job(self, kind: str) -> dict:
        if self._job['running']:
            return {'ok': False, 'error': 'A data update is already running.'}
        cmds = {'leagues': [sys.executable, os.path.join(ROOT, 'scripts', 'fetch_latest.py'),
                            '--apply', '--refresh-processed'],
                'internationals': [sys.executable, os.path.join(ROOT, 'scripts', 'international.py'),
                                   'update']}
        if kind not in cmds:
            return {'ok': False, 'error': f'unknown job {kind!r}'}
        self._job = {'running': True, 'kind': kind, 'log': '', 'ok': None,
                     'started': time.time(), 'finished': None}

        def work():
            try:
                proc = subprocess.Popen(cmds[kind], cwd=ROOT, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True)
                for line in proc.stdout:
                    self._job['log'] = (self._job['log'] + line)[-20000:]
                rc = proc.wait()
                # fetch_latest exits 3 when some seasons are simply not published
                self._job['ok'] = rc in (0, 3)
                if self._job['ok']:
                    toto._INTL_CACHE = None
                    self.load()                      # pick up the new data
            except Exception as e:
                self._job['log'] += f'\n{type(e).__name__}: {e}'
                self._job['ok'] = False
            finally:
                self._job['running'] = False
                self._job['finished'] = time.time()

        threading.Thread(target=work, daemon=True).start()
        return {'ok': True}

    def job(self) -> dict:
        return dict(self._job)


ENGINE = Engine()
