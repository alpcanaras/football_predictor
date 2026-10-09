"""
Engine behind the web app
=========================
Holds the loaded predictor and the app's state, and exposes plain functions —
JSON-ready dicts in, dicts out. Nothing here re-implements a model or the Toto
maths: numbers come from scripts/predictor.py (one best model per task),
tickets from scripts/ticket.py, history from scripts/track.py, names from
scripts/names.py.

One coupon, no game switch. Both pools pay a prize for at most three wrong
(13er Wette: 10+ of 13, Spor Toto: 12+ of 15), so the coupon is simply "the
matches you added" and its size decides what it is: 13 plays as the 13er
Wette, 15 as Spor Toto, anything else is previewed with the same rule.

The coupon lives in <home>/data/_toto/coupon.txt in the one-line-per-match
format ("Home - Away  1.95 3.40 3.90"); what that format cannot hold (crowd %,
locked symbols, budget) lives next to it in coupon.meta.json.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
import time
import traceback

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts import config, names, toto  # noqa: E402
from scripts import ticket as ticket_mod  # noqa: E402
from scripts.predictor import Predictor  # noqa: E402

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
UPDATE_EVERY_H = 12                    # the packaged app refreshes data this often
SRC_LABEL = {'odds': 'odds', 'model': 'model', 'national': 'nations', 'euro': 'club elo'}


def game_for(n: int) -> dict:
    """What a coupon of n matches is."""
    for size, label, flag, prize in GAMES:
        if n == size:
            return {'n': n, 'label': label, 'flag': flag, 'prize_from': prize,
                    'complete': True, 'key': 'german' if size == 13 else 'turkish'}
    return {'n': n, 'label': None, 'flag': '', 'complete': False,
            'prize_from': max(1, n - ALLOWED_WRONG),
            'key': 'german' if n <= 13 else 'turkish'}


def _atomic_write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(text)
    os.replace(tmp, path)


def row_key(home: str, away: str) -> str:
    """Order-insensitive, accent-folded identity of a fixture."""
    return '|'.join(sorted((names.norm(home), names.norm(away))))


class _Log:
    """A write-only file that appends to the running job's log."""

    def __init__(self, job: dict):
        self.job = job

    def write(self, text):
        self.job['log'] = (self.job['log'] + text)[-20000:]
        return len(text)

    def flush(self):
        pass


class Engine:
    def __init__(self):
        self.state, self.step, self.error = 'starting', '', None
        self.pred: Predictor | None = None
        self.teams: list[dict] = []
        self.loaded_at = None
        self.fx = {'state': 'idle', 'done': 0, 'total': 0, 'rows': [], 'loaded_at': None, 'error': None}
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
            if self.pred is None:
                self.pred = Predictor()
            else:
                self.pred.reload()
            self.step = 'Loading models'
            self.pred.bundles()
            self.step = 'Team names'
            self.teams = self._team_list()
            self.fx = {'state': 'idle', 'done': 0, 'total': 0, 'rows': [], 'loaded_at': None, 'error': None}
            self.loaded_at = time.time()
            self.state, self.step = 'ready', ''
            self.start_fixtures()                 # prices the week ahead in the background
            self._maybe_auto_update()
        except Exception as e:                           # shown on the page
            self.state, self.error = 'error', f'{type(e).__name__}: {e}'
            traceback.print_exc()

    def start(self) -> None:
        threading.Thread(target=self.load, daemon=True).start()

    def need_ready(self):
        if self.state != 'ready':
            raise RuntimeError('Still warming up — try again in a few seconds.')

    def _team_list(self) -> list[dict]:
        idx = self.pred.index
        out = []
        for canon, (kind, lg) in idx.entries.items():
            out.append({'name': canon, 'display': idx.display_of[canon], 'kind': kind,
                        'league': lg,
                        'league_name': config.LEAGUE_REGISTRY.get(lg, {}).get('display_name', lg)
                        if kind == 'club' else 'National team',
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

    def save_settings(self, **changes) -> dict:
        s = self.settings()
        s.update({k: v for k, v in changes.items() if v is not None})
        _atomic_write(SETTINGS_FILE, json.dumps(s, indent=2))
        return s

    def auto_update_on(self) -> bool:
        default = os.environ.get('FOOTBALL_PREDICTOR_AUTOUPDATE') == '1'
        return bool(self.settings().get('auto_update', default))

    # ---------------------------------------------------------------- status
    def status(self) -> dict:
        s = self.settings()
        out = {'state': self.state, 'step': self.step, 'error': self.error,
               'allowed_wrong': ALLOWED_WRONG,
               'games': [{'n': n, 'label': lab, 'flag': flag, 'prize_from': pf}
                         for n, lab, flag, pf in GAMES],
               'job': {k: v for k, v in self._job.items() if k != 'log'},
               'auto_update': self.auto_update_on(), 'last_update': s.get('last_update'),
               'version': os.environ.get('FOOTBALL_PREDICTOR_VERSION', 'dev')}
        if self.pred:
            today = pd.Timestamp.today().normalize()
            last = self.pred.last_result
            out['data_through'] = str(self.pred.data_through.date())
            stale = sorted(((int((today - d).days), lg) for lg, d in last.items()
                            if (today - d).days > 28), reverse=True)
            out['stale'] = [{'league': lg, 'name': config.LEAGUE_REGISTRY.get(lg, {}).get('display_name', lg),
                             'days': d} for d, lg in stale]
            out['n_leagues'] = len(last)
            out['n_teams'] = len(self.pred.clubs)
            try:
                out['models'] = self.pred.info()
            except Exception as e:
                out['models'] = {'ok': False, 'error': str(e)}
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
        return {'ok': self.pred.index.remember(typed, canonical)}

    # ------------------------------------------------------------- analysis
    @staticmethod
    def _slim(r: dict) -> dict:
        """The part of a prediction a coupon row or a list row needs."""
        return {
            'probs': r.get('p1x2'), 'src': SRC_LABEL.get(r.get('src'), r.get('src')) if r.get('ok') else 'none',
            'market': r.get('market_1x2'), 'model': r.get('model_1x2'),
            'ou25': (r.get('ou') or {}).get('2.5'), 'ou25_src': r.get('ou_src'),
            'btts': r.get('btts'), 'xg': r.get('xg'),
            'score': (r.get('scores') or [{}])[0].get('score'),
            'score_p': (r.get('scores') or [{}])[0].get('p'),
        }

    def analyze_row(self, home: str, away: str, odds=None) -> dict:
        """Numbers for one coupon row, where they came from, and name
        suggestions for anything unrecognised."""
        self.need_ready()
        home, away = str(home or '').strip(), str(away or '').strip()
        if not home or not away:
            return {'home': home, 'away': away, 'probs': None, 'src': 'none', 'sides': {'home': None, 'away': None}}
        o = list(odds) if odds and len(odds) == 3 and all(x and float(x) > 1 for x in odds) else None
        r = self.pred.predict(home, away, odds=o)
        out = {'home': r['home'], 'away': r['away'], 'sides': r['sides'],
               'league': r.get('league'), 'league_name': r.get('league_name', ''), **self._slim(r)}
        if not r['ok']:
            out.update(probs=None, src='none')
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
        with self._fx_lock:
            fx = self.fx
            if fx['state'] == 'loading':
                return
            fresh = fx['state'] == 'ready' and time.time() - (fx['loaded_at'] or 0) < FIXTURE_TTL
            if fresh and not force and not refetch:
                return
            self.fx = {'state': 'loading', 'done': 0, 'total': 0, 'rows': [],
                       'loaded_at': None, 'error': None, 'started': time.time()}
        threading.Thread(target=self._price_fixtures, args=(refetch,), daemon=True).start()

    def _price_fixtures(self, refetch: bool) -> None:
        fx = self.fx
        try:
            from scripts import fixtures as fx_mod
            try:
                fx_mod.fetch(force=refetch)
            except Exception:                       # offline: use the saved feed
                traceback.print_exc()
            feed = fx_mod.load(fetch_if_missing=False)
            now = pd.Timestamp.now().normalize()
            feed = feed[(feed['Date'] >= now) & (feed['Date'] <= now + pd.Timedelta(days=FIXTURE_DAYS))]
            feed = feed.dropna(subset=['league', 'HomeTeam', 'AwayTeam', 'Date'])
            feed = feed.drop_duplicates(['league', 'Date', 'HomeTeam', 'AwayTeam'])
            feed = feed.sort_values(['Date', 'Time', 'league', 'HomeTeam'], na_position='last')
            idx = self.pred.index
            ok = lambda x: pd.notna(x) and x > 1                                       # noqa: E731
            base = []
            for m in feed.to_dict('records'):
                odds = [float(m[c]) for c in ('OddsH', 'OddsD', 'OddsA')] \
                    if all(ok(m[c]) for c in ('OddsH', 'OddsD', 'OddsA')) else None
                ou = [float(m['OddsOver25']), float(m['OddsUnder25'])] \
                    if ok(m['OddsOver25']) and ok(m['OddsUnder25']) else None
                home, away, lg = m['HomeTeam'], m['AwayTeam'], m['league']
                base.append({
                    'id': f'{lg}|{m["Date"].date()}|{home}|{away}', 'date': str(m['Date'].date()),
                    'time': (str(m['Time']) if pd.notna(m.get('Time')) else None),
                    'league': lg, 'league_name': config.LEAGUE_REGISTRY.get(lg, {}).get('display_name', lg),
                    'home': home, 'away': away,
                    'home_display': idx.display_of.get(home, home), 'away_display': idx.display_of.get(away, away),
                    'odds': odds, 'ou_odds': ou, 'probs': None, 'src': 'odds' if odds else None,
                    'ou25': None, 'btts': None, 'score': None, 'pending': True})
            fx['rows'] = base
            fx['total'] = len(base)
            try:                                       # build league states in parallel first
                from scripts import v2_bundle
                v2_bundle.warm(sorted({r['league'] for r in base}))
            except Exception:
                traceback.print_exc()
            for i, r in enumerate(base):
                new = dict(r, pending=False)
                try:
                    p = self.pred.predict(r['home'], r['away'], odds=r['odds'], ou_odds=r['ou_odds'],
                                          date=pd.Timestamp(r['date']))
                    if p['ok']:
                        new.update(self._slim(p), xg=p.get('xg'))
                except Exception:
                    traceback.print_exc()           # leaves the row without numbers
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
                'rows': list(fx['rows']), 'error': fx.get('error'), 'loaded_at': fx['loaded_at']}

    # ----------------------------------------------------------------- match
    def match(self, home: str, away: str, odds=None, ou_odds=None) -> dict:
        """The full card for any pair of teams."""
        self.need_ready()
        o = list(odds) if odds and len(odds) == 3 and all(x and float(x) > 1 for x in odds) else None
        r = self.pred.predict(home, away, odds=o, ou_odds=ou_odds)
        r['src'] = SRC_LABEL.get(r.get('src'), r.get('src')) if r['ok'] else 'none'
        return r

    # --------------------------------------------------------------- history
    def history(self) -> dict:
        from scripts import track
        graded = track.grade_all(club=self.pred.results if self.pred else None)
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
    def _maybe_auto_update(self) -> None:
        if not self.auto_update_on() or self._job['running']:
            return
        last = self.settings().get('last_update')
        if last is None or time.time() - float(last) > UPDATE_EVERY_H * 3600:
            self.start_job('all')

    def start_job(self, kind: str) -> dict:
        """Download the newest results (football-data.co.uk) and the
        international results, in this process, and reload everything."""
        if self._job['running']:
            return {'ok': False, 'error': 'A data update is already running.'}
        if kind not in ('all', 'leagues', 'internationals'):
            return {'ok': False, 'error': f'unknown job {kind!r}'}
        self._job = {'running': True, 'kind': kind, 'log': '', 'ok': None,
                     'started': time.time(), 'finished': None}

        def work():
            log = _Log(self._job)
            ok = True
            try:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    if kind in ('all', 'leagues'):
                        from scripts import fetch_latest
                        rc = fetch_latest.main(['--apply'])
                        print(f'\n  league data: exit {rc} (3 = some seasons not published yet, normal)')
                        ok = ok and rc in (0, 3)
                    if kind in ('all', 'internationals'):
                        from scripts import international as intl
                        intl.cmd_update(None)
                if ok:
                    self.save_settings(last_update=time.time())
                    self.load()                      # reread data, then re-price
                    self.start_fixtures(force=True, refetch=True)
            except Exception as e:
                self._job['log'] += f'\n{type(e).__name__}: {e}'
                ok = False
            finally:
                self._job['ok'] = ok
                self._job['running'] = False
                self._job['finished'] = time.time()

        threading.Thread(target=work, daemon=True).start()
        return {'ok': True}

    def job(self) -> dict:
        return dict(self._job)


ENGINE = Engine()
