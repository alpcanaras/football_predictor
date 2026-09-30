"""
Engine behind the web app
=========================
Loads the heavy context once (match data, team tables, national-team model)
and exposes plain functions — JSON-ready dicts in, dicts out — over the
existing modules. Nothing here re-implements a model or the Toto maths:
probabilities come from scripts/toto.match_probs (odds > club model v1/v2 >
European clubs > national teams), tickets from scripts/ticket.build, history
from scripts/track.

Coupons stay in data/_toto/<game>.txt in the same one-line-per-match format
the Streamlit app uses ("Home - Away  1.95 3.40 3.90"), so both apps share
them. What the text format cannot hold — crowd %, locked symbols, budget,
objective — lives next to it in <game>.meta.json.
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

from scripts import config, toto  # noqa: E402
from scripts import predict as predict_mod  # noqa: E402
from scripts import ticket as ticket_mod  # noqa: E402

COUPON_DIR = os.environ.get('FOOTBALL_PREDICTOR_TOTO_DIR',
                            os.path.join(config.DATA_DIR, '_toto'))
SETTINGS_FILE = os.path.join(COUPON_DIR, 'settings.json')
GAME_INFO = {
    'turkish': {'label': 'Spor Toto', 'flag': '🇹🇷'},
    'german': {'label': '13er Wette', 'flag': '🇩🇪'},
}
# Model calls touch shared caches (v2 per-league states, the intl model) that
# are not safe to fill from two threads at once; one user, so serialise.
MODEL_LOCK = threading.RLock()


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
    return '|'.join(sorted((toto._fold(home), toto._fold(away))))


class Engine:
    def __init__(self):
        self.state, self.step, self.error = 'starting', '', None
        self.ctx = None
        self.teams: list[dict] = []
        self.loaded_at = None
        self._fixtures_cache: dict = {}
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
            self.step = 'Settings'
            predict_mod.set_model_mode(self.settings().get('model_mode', 'auto'))
            teams = [{'name': t, 'league': lg, 'league_name': _disp(lg), 'kind': 'club'}
                     for t, lg in t2l.items()]
            teams += [{'name': n, 'league': None, 'league_name': 'National team',
                       'kind': 'national'} for n in ic['ratings']]
            teams.sort(key=lambda t: (t['kind'] != 'club', t['name'].lower()))
            last = hist.groupby('league')['Date'].max()
            self.ctx = {'hist': hist, 'team_stats': team_stats,
                        'team_to_league': t2l, 'teams': set(t2l),
                        'weights': toto._load_blend_weights(),
                        'last_result': {lg: pd.Timestamp(d) for lg, d in last.items()}}
            self.teams = teams
            self._fixtures_cache.clear()
            self.loaded_at = time.time()
            self.state, self.step = 'ready', ''
        except Exception as e:                           # shown on the page
            self.state, self.error = 'error', f'{type(e).__name__}: {e}'
            traceback.print_exc()

    def start(self) -> None:
        threading.Thread(target=self.load, daemon=True).start()

    def need_ready(self):
        if self.state != 'ready':
            raise RuntimeError('Still warming up — try again in a few seconds.')

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
        self._fixtures_cache.clear()
        return self.status()

    # ---------------------------------------------------------------- status
    def status(self) -> dict:
        out = {'state': self.state, 'step': self.step, 'error': self.error,
               'model_mode': predict_mod.get_model_mode(),
               'v2_auto_leagues': sorted(predict_mod.V2_AUTO_LEAGUES),
               'games': {g: {'n': n, 'threshold': t, 'top': top, **GAME_INFO.get(g, {})}
                         for g, (n, t, top) in toto.GAMES.items()},
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
    def _paths(self, game: str):
        if game not in toto.GAMES:
            raise ValueError(f'unknown game {game!r}')
        return (os.path.join(COUPON_DIR, f'{game}.txt'),
                os.path.join(COUPON_DIR, f'{game}.meta.json'))

    def load_coupon(self, game: str) -> dict:
        txt, meta_p = self._paths(game)
        try:
            with open(txt, encoding='utf-8') as f:
                text = f.read()
        except OSError:
            text = ''
        try:
            with open(meta_p, encoding='utf-8') as f:
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
        n, thr, top = toto.GAMES[game]
        return {'game': game, 'n': n, 'threshold': thr, 'top': top, 'rows': rows,
                'budget': int(meta.get('budget', 1)),
                'tilt': float(meta.get('tilt', ticket_mod.DEFAULT_TILT)),
                'objective': meta.get('objective', 'hit'),
                'updated': meta.get('updated')}

    def save_coupon(self, game: str, body: dict) -> dict:
        txt, meta_p = self._paths(game)
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
        _atomic_write(txt, '\n'.join(lines))
        meta = {'rows': per, 'budget': int(body.get('budget', 1)),
                'tilt': float(body.get('tilt', ticket_mod.DEFAULT_TILT)),
                'objective': body.get('objective', 'hit'),
                'updated': pd.Timestamp.now().isoformat(timespec='seconds')}
        _atomic_write(meta_p, json.dumps(meta, indent=2, ensure_ascii=False))
        return {'ok': True, 'n': len(lines), 'updated': meta['updated']}

    def parse_text(self, text: str) -> list[dict]:
        out = []
        for _, r in toto.parse_lines(text).iterrows():
            odds = [float(r[c]) for c in ('o1', 'ox', 'o2')] \
                if all(pd.notna(r.get(c)) for c in ('o1', 'ox', 'o2')) else None
            out.append({'home': str(r['home']), 'away': str(r['away']), 'odds': odds})
        return out

    # ------------------------------------------------------------- analysis
    def _known(self, name: str, ic: dict) -> str | None:
        """How a coupon name resolves: 'club', 'national', 'euro' or None."""
        if toto._fuzzy(name, self.ctx['teams']):
            return 'club'
        if toto._intl_name(name, ic):
            return 'national'
        try:
            from scripts import clubs_europe
            if clubs_europe.resolve(name):
                return 'euro'
        except Exception:
            pass
        return None

    def analyze_row(self, home: str, away: str, odds=None) -> dict:
        """Probabilities for one coupon row, where they came from, both club
        model versions, and name suggestions for anything unrecognised."""
        self.need_ready()
        home, away = str(home or '').strip(), str(away or '').strip()
        out = {'home': home, 'away': away, 'probs': None, 'src': 'none',
               'v1': None, 'v2': None, 'market': None, 'league': None,
               'league_name': '', 'known': {'home': None, 'away': None},
               'suggest': {'home': [], 'away': []}}
        if not home or not away:
            return out
        ic = toto._load_intl()
        for side, name in (('home', home), ('away', away)):
            out['known'][side] = self._known(name, ic)
            if out['known'][side] is None:
                out['suggest'][side] = toto.suggest_names(
                    name, toto.all_known_names(self.ctx['team_to_league']), 5)
        o = list(odds) if odds and len(odds) == 3 and all(x and float(x) > 1 for x in odds) else None
        row = pd.Series({'home': home, 'away': away,
                         'o1': o[0] if o else None, 'ox': o[1] if o else None,
                         'o2': o[2] if o else None})
        ctx = dict(self.ctx)
        ctx['model_detail'] = {}
        with MODEL_LOCK:
            p, src, model = toto.match_probs(row, ctx)
        det = ctx['model_detail'].get((home, away))
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
                out['src'] = 'feed' if det['feed_odds'] else det['version']
            else:
                out['src'] = src                         # euro / intl
        return out

    # --------------------------------------------------------------- tickets
    @staticmethod
    def effective_threshold(game: str, n_rows: int) -> int:
        """The prize rule as a number of allowed misses (Spor Toto: 3 of 15),
        so a coupon still being filled in previews the same tolerance."""
        n, thr, _ = toto.GAMES[game]
        return max(1, n_rows - (n - thr))

    @staticmethod
    def _tiers(probs, masks, threshold, top):
        qs = [float(sum(p[o] for o in range(3) if (m >> o) & 1)) for p, m in zip(probs, masks)]
        d = toto.pb_distribution(qs)
        return ({str(k): float(d[k:].sum()) if k < len(d) else 0.0
                 for k in range(threshold, top + 1)}, float(sum(qs)))

    def ticket(self, body: dict) -> dict:
        game = body.get('game', 'turkish')
        n, thr, top = toto.GAMES[game]
        probs = [np.asarray(p, float) for p in body['probs']]
        if not probs:
            return {'labels': [], 'masks': [], 'columns': 0}
        thr_eff = self.effective_threshold(game, len(probs))
        top_eff = len(probs)
        budget = max(1, int(body.get('budget', 1)))
        inv = {v: k for k, v in ticket_mod.MASK_LABEL.items()}
        fixed = {i: inv[l] for i, l in enumerate(body.get('locks') or []) if l in inv}
        crowd = body.get('crowd')
        use_crowd = (body.get('objective') == 'crowd' and crowd
                     and len(crowd) == len(probs) and all(c and len(c) == 3 for c in crowd))
        t0 = time.time()
        r = ticket_mod.build(probs, thr_eff, budget,
                             crowd=np.asarray(crowd, float) if use_crowd else None,
                             tilt=float(body.get('tilt', ticket_mod.DEFAULT_TILT)),
                             fixed=fixed, n_sims=12000, restarts=4)
        tiers, exp = self._tiers(probs, r['masks'], thr_eff, top_eff)
        single = [1 << int(np.argmax(p)) for p in probs]
        s_tiers, s_exp = self._tiers(probs, single, thr_eff, top_eff)
        return {'labels': r['labels'], 'masks': r['masks'], 'columns': r['columns'],
                'budget': budget, 'objective': r['objective'],
                'over_budget': r.get('over_budget', False),
                'threshold': thr_eff, 'full_threshold': thr, 'tiers': tiers,
                'p_prize': tiers[str(thr_eff)], 'expected': exp,
                'single': {'p_prize': s_tiers[str(thr_eff)], 'expected': s_exp},
                'n_double': sum(1 for m in r['masks'] if ticket_mod.popcount(m) == 2),
                'n_triple': sum(1 for m in r['masks'] if ticket_mod.popcount(m) == 3),
                'ms': int((time.time() - t0) * 1000)}

    def ladder(self, body: dict) -> list[dict]:
        """What each extra doubling of the budget buys (hit objective)."""
        game = body.get('game', 'turkish')
        probs = [np.sort(np.asarray(p, float))[::-1] for p in body['probs']]
        thr = self.effective_threshold(game, len(probs))
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
        game = body.get('game', 'turkish')
        probs = [np.asarray(p, float) for p in body['probs']]
        thr = self.effective_threshold(game, len(probs))
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
    def fixtures(self, days: int = 3) -> dict:
        self.need_ready()
        key = (days, predict_mod.get_model_mode())
        hit = self._fixtures_cache.get(key)
        if hit and time.time() - hit[0] < 600:
            return hit[1]
        from scripts import today as today_mod
        with MODEL_LOCK:
            rows = today_mod.club_section(days, self.ctx['hist'], self.ctx['team_stats'],
                                          self.ctx['team_to_league'])
        out = []
        for r in rows:
            out.append({
                'date': str(r['Date']), 'league': r['_league'], 'league_name': r['League'],
                'home': r['Home'], 'away': r['Away'],
                'v1': list(r['_v1']) if r.get('_v1') else None,
                'v2': list(r['_v2']) if r.get('_v2') else None,
                'market': list(r['_mkt']) if r.get('_mkt') else None,
                'odds': [float(x) for x in r['Odds(1/X/2)'].split('/')] if r.get('Odds(1/X/2)') else None,
                'goals_v1': r.get('_g1'), 'goals_v2': r.get('_g2'),
                'ou25': r.get('P(O2.5)'), 'btts': r.get('P(BTTS)'),
            })
        res = {'fixtures': out, 'mode': predict_mod.get_model_mode(),
               'loaded_at': pd.Timestamp.now().isoformat(timespec='seconds')}
        self._fixtures_cache[key] = (time.time(), res)
        return res

    # ----------------------------------------------------------------- match
    def match(self, home: str, away: str) -> dict:
        """Full card for any pair: club model (both versions + goals) when the
        clubs share a league, else the European / national fallback."""
        self.need_ready()
        t2l = self.ctx['team_to_league']
        h = toto._fuzzy(home, self.ctx['teams'])
        a = toto._fuzzy(away, self.ctx['teams'])
        if h and a and t2l.get(h) == t2l.get(a):
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
            return {'kind': 'club', 'home': h, 'away': a, 'league': lg, 'league_name': _disp(lg),
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
        return {'kind': r['src'], 'home': home, 'away': away, 'used': r['probs'],
                'known': r['known'], 'suggest': r['suggest'], 'league_name': r['league_name']}

    # --------------------------------------------------------- internationals
    def internationals(self, days: int = 14) -> dict:
        self.need_ready()
        from scripts import international as intl
        rows = []
        try:
            fx = pd.read_csv(intl.WC_FIXTURES_FILE, parse_dates=['date'])
        except (OSError, ValueError):
            fx = pd.DataFrame()
        if not fx.empty:
            now = pd.Timestamp.now().normalize()
            fx = fx[(fx['date'] >= now) & (fx['date'] <= now + pd.Timedelta(days=days))]
            ic = toto._load_intl()
            for _, m in fx.sort_values('date').iterrows():
                hm, aw = m['home_team'], m['away_team']
                if hm not in ic['ratings'] or aw not in ic['ratings']:
                    continue
                pr = ic['model'].market_probs(ic['ratings'][hm], ic['ratings'][aw],
                                              neutral=bool(m.get('neutral', False)))
                rows.append({'date': str(m['date'].date()), 'home': hm, 'away': aw,
                             'tournament': m.get('tournament', ''),
                             'probs': [pr['p_home'], pr['p_draw'], pr['p_away']],
                             'ou25': pr.get('p_over25'), 'btts': pr.get('p_btts')})
        return {'fixtures': rows, 'results_through': str(intl.load_results()['date'].max().date())}

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
        game = body['game']
        _, thr, _ = toto.GAMES[game]
        matches = [{'home': m['home'], 'away': m['away'], 'p1': m['probs'][0],
                    'px': m['probs'][1], 'p2': m['probs'][2], 'pick': m['pick'],
                    'src': m.get('src')} for m in body['matches']]
        e = track.save_coupon(game, matches, budget=int(body.get('budget', 1)),
                              threshold=thr, p_threshold=float(body['p_prize']),
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
