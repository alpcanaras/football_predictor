#!/usr/bin/env python3
"""
Coupon history + calibration tracker
====================================
Records the coupons you actually analysed and grades them once the results are
in, so "does this thing work?" becomes a number instead of a feeling.

Each saved coupon keeps, per match, the probabilities the model gave at the
time and the pick it implied. Grading looks the results up in the processed
club data and the international results file, then compares what happened with
what was predicted:

  * hit rate      — how often you cleared the prize threshold
  * correct vs expected — the Poisson-binomial mean is the honest benchmark;
    consistently landing below it means the probabilities are optimistic
  * calibration   — of all the picks given ~70% confidence, how many landed

    python scripts/track.py list
    python scripts/track.py grade
    python scripts/track.py calibration
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from scripts import config, toto

OUTCOMES = ['1', 'X', '2']
_FTR_TO_PICK = {'H': '1', 'D': 'X', 'A': '2'}


def history_dir() -> str:
    return os.environ.get(
        'FOOTBALL_PREDICTOR_TOTO_DIR',
        os.path.join(config.DATA_DIR, '_toto'))


def history_file() -> str:
    return os.path.join(history_dir(), 'history.jsonl')


# =============================================================================
# WRITE
# =============================================================================
def save_coupon(game: str, matches: list[dict], budget: int = 1,
                threshold: int | None = None, p_threshold: float | None = None,
                note: str = '') -> dict:
    """Append one analysed coupon to the history.

    `matches` items need: home, away, p1, px, p2, pick (and optionally src).
    """
    entry = {
        'id': dt.datetime.now().strftime('%Y%m%d-%H%M%S'),
        'saved_at': dt.datetime.now().isoformat(timespec='seconds'),
        'game': game,
        'budget': int(budget),
        'threshold': threshold,
        'p_threshold': p_threshold,
        'note': note,
        'matches': matches,
    }
    os.makedirs(history_dir(), exist_ok=True)
    with open(history_file(), 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    return entry


def load_history() -> list[dict]:
    path = history_file()
    if not os.path.isfile(path):
        return []
    out = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def delete_coupon(coupon_id: str) -> bool:
    rows = [e for e in load_history() if e.get('id') != coupon_id]
    if len(rows) == len(load_history()):
        return False
    with open(history_file(), 'w', encoding='utf-8') as f:
        for e in rows:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')
    return True


# =============================================================================
# GRADE
# =============================================================================
class ResultLookup:
    """Finds the actual 1X2 outcome of a match, club or national."""

    def __init__(self, club: pd.DataFrame | None = None,
                 intl: pd.DataFrame | None = None):
        if club is None:
            club = pd.read_csv(config.PROCESSED_DATA_FILE,
                               usecols=['Date', 'HomeTeam', 'AwayTeam', 'FTR'],
                               parse_dates=['Date'])
        self.club = club
        self._club_idx = {}
        for h, a, r, d in zip(club['HomeTeam'], club['AwayTeam'],
                              club['FTR'], club['Date']):
            self._club_idx.setdefault(
                (toto._fold(h), toto._fold(a)), []).append((d, r))

        self._intl_idx = {}
        if intl is None:
            try:
                from scripts import international as intl_mod
                intl = intl_mod.load_results()
            except Exception:
                intl = None
        if intl is not None:
            for h, a, hs, as_, d in zip(intl['home_team'], intl['away_team'],
                                        intl['home_score'], intl['away_score'],
                                        intl['date']):
                r = 'H' if hs > as_ else ('D' if hs == as_ else 'A')
                self._intl_idx.setdefault(
                    (toto._fold(h), toto._fold(a)), []).append((d, r))

    # A coupon is normally saved shortly before kickoff, but may be saved just
    # after; either way we want *that* meeting, not last season's.
    BACK_DAYS = 7
    FORWARD_DAYS = 90

    def lookup(self, home: str, away: str, after=None):
        """The outcome of the meeting this coupon refers to ('H'/'D'/'A')."""
        key = (toto._fold(home), toto._fold(away))
        hits = self._club_idx.get(key, []) + self._intl_idx.get(key, [])
        if not hits:
            return None
        if after is not None:
            ref = pd.Timestamp(after).normalize()
            lo = ref - pd.Timedelta(days=self.BACK_DAYS)
            hi = ref + pd.Timedelta(days=self.FORWARD_DAYS)
            window = [(d, r) for d, r in hits
                      if lo <= pd.Timestamp(d) <= hi]
            if not window:
                return None
            # the first meeting in the window is the one that was coupled
            return min(window, key=lambda t: pd.Timestamp(t[0]))[1]
        return max(hits, key=lambda t: pd.Timestamp(t[0]))[1]


def grade_coupon(entry: dict, lookup: ResultLookup) -> dict:
    """Compare a saved coupon with what actually happened."""
    saved = entry.get('saved_at')
    rows, correct, graded = [], 0, 0
    exp = 0.0
    for m in entry.get('matches', []):
        probs = [m.get('p1'), m.get('px'), m.get('p2')]
        pick = m.get('pick')
        top = max(p for p in probs if p is not None) if any(
            p is not None for p in probs) else None
        actual_ftr = lookup.lookup(m['home'], m['away'], after=saved)
        actual = _FTR_TO_PICK.get(actual_ftr) if actual_ftr else None
        ok = None
        if actual is not None:
            graded += 1
            ok = (actual == pick)
            correct += int(ok)
        if top is not None:
            exp += float(top)
        rows.append({**m, 'actual': actual, 'ok': ok})
    return {
        'id': entry.get('id'), 'game': entry.get('game'),
        'saved_at': saved, 'threshold': entry.get('threshold'),
        'p_threshold': entry.get('p_threshold'),
        'n': len(entry.get('matches', [])), 'graded': graded,
        'correct': correct, 'expected': exp, 'rows': rows,
        'complete': graded == len(entry.get('matches', [])) and graded > 0,
    }


def grade_all() -> list[dict]:
    hist = load_history()
    if not hist:
        return []
    lookup = ResultLookup()
    return [grade_coupon(e, lookup) for e in hist]


# =============================================================================
# CLI
# =============================================================================
def cmd_list(_args):
    hist = load_history()
    if not hist:
        print("  No saved coupons yet — analyse one in the app and hit "
              "'Save to history'.")
        return
    print(f"\n  {len(hist)} saved coupon(s)\n")
    print(f"  {'id':<16} {'game':<9} {'matches':>7} {'budget':>7} "
          f"{'P(prize)':>9}  note")
    print('  ' + '-' * 64)
    for e in hist:
        p = e.get('p_threshold')
        print(f"  {e['id']:<16} {e['game']:<9} {len(e['matches']):>7} "
              f"{e.get('budget', 1):>7} "
              f"{(f'{p:.1%}' if p else '-'):>9}  {e.get('note', '')[:24]}")


def cmd_grade(_args):
    graded = grade_all()
    if not graded:
        print("  Nothing to grade yet.")
        return
    print(f"\n  {'id':<16} {'game':<9} {'result':>9} {'expected':>9} "
          f"{'prize':>7}")
    print('  ' + '-' * 56)
    hits = done = 0
    tot_correct = tot_expected = 0.0
    for g in graded:
        if not g['graded']:
            print(f"  {g['id']:<16} {g['game']:<9} {'pending':>9}")
            continue
        thr = g['threshold']
        won = thr is not None and g['correct'] >= thr
        mark = '✅' if won else '—'
        if g['complete']:
            done += 1
            hits += int(won)
            tot_correct += g['correct']
            tot_expected += g['expected']
        score = f"{g['correct']}/{g['graded']}"
        print(f"  {g['id']:<16} {g['game']:<9} {score:>9} "
              f"{g['expected']:>9.1f} {mark:>7}")
    if done:
        print('  ' + '-' * 56)
        print(f"  {done} completed coupon(s): {hits} cleared the threshold "
              f"({hits/done:.0%})")
        print(f"  correct {tot_correct:.0f} vs expected {tot_expected:.1f} "
              f"({tot_correct - tot_expected:+.1f})")
        if tot_correct < tot_expected - 2:
            print("  -> landing below expectation: the probabilities are "
                  "running optimistic.")


def cmd_calibration(_args):
    graded = grade_all()
    rows = [r for g in graded for r in g['rows'] if r.get('ok') is not None]
    if len(rows) < 10:
        print(f"  Only {len(rows)} graded picks so far — need a few more "
              "coupons before calibration means anything.")
        return
    conf = np.array([max(r['p1'], r['px'], r['p2']) for r in rows])
    hit = np.array([bool(r['ok']) for r in rows])
    print(f"\n  Calibration over {len(rows)} graded picks\n")
    print(f"  {'confidence':<16} {'n':>5} {'predicted':>10} {'actual':>8}")
    print('  ' + '-' * 44)
    for lo, hi in [(0, .4), (.4, .5), (.5, .6), (.6, .7), (.7, .8), (.8, 1.01)]:
        m = (conf >= lo) & (conf < hi)
        if m.sum() >= 3:
            print(f"  {f'{lo:.0%}-{hi:.0%}':<16} {m.sum():>5} "
                  f"{conf[m].mean():>10.1%} {hit[m].mean():>8.1%}")
    print('  ' + '-' * 44)
    print(f"  {'overall':<16} {len(rows):>5} {conf.mean():>10.1%} "
          f"{hit.mean():>8.1%}")


def main():
    ap = argparse.ArgumentParser(description='Toto coupon history + grading')
    sub = ap.add_subparsers(dest='command')
    sub.add_parser('list', help='List saved coupons')
    sub.add_parser('grade', help='Grade saved coupons against results')
    sub.add_parser('calibration', help='Confidence vs actual hit rate')
    args = ap.parse_args()
    {'list': cmd_list, 'grade': cmd_grade,
     'calibration': cmd_calibration}.get(args.command or 'list', cmd_list)(args)


if __name__ == '__main__':
    main()
