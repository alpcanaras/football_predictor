#!/usr/bin/env python3
"""
Ticket maker
============
Turns a coupon plus a budget into the actual ticket you fill in: for every
match, the exact symbols to play (1, X, 2, 1X, 12, X2, 1X2).

Two objectives:

  * hit    - maximise P(>= threshold correct). What you want if you only care
             about winning *something*.
  * payout - maximise expected return, which is NOT the same thing. Everyone
             with 12+ shares the pot, so the prize depends on how many others
             also got there. Back the field's favourites and you win in the
             weeks thousands of others win too. The weeks worth winning are
             the ones the crowd loses, so given crowd percentages this leans
             toward outcomes you rate higher than the field does.

Crowd percentages are the share of players on each outcome (the "oynanma
yuzdesi"). Hand-enter them as three numbers per match; without them the
engine falls back to the hit objective.

    python scripts/ticket.py --coupon coupon.csv --game turkish --budget 100
    python scripts/ticket.py --coupon coupon.csv --budget 100 --objective payout
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from scripts import toto

SYMBOLS = ('1', 'X', '2')
# bitmask -> ticket notation. bit0 = '1', bit1 = 'X', bit2 = '2'
MASK_LABEL = {0b001: '1', 0b010: 'X', 0b100: '2',
              0b011: '1X', 0b101: '12', 0b110: 'X2', 0b111: '1X2'}
ALL_MASKS = tuple(MASK_LABEL)
# How many coupons the field plays. Only the order of magnitude matters: it
# sets how much a contrarian week is actually worth. Override with --field.
DEFAULT_FIELD = 200_000
# How hard to lean into unpopular outcomes. Pure expected-value maximisation
# (tilt=1) is mathematically defensible and practically ruinous: on a real
# coupon it cut the hit rate from 21% to 1% — once every 85 weeks — to chase
# weeks when the field is wiped out. Measured sweeps put the cliff around
# 0.75; below it you buy ~20-30% more expected payout for a few points of hit
# rate, which is a trade a human can actually run.
DEFAULT_TILT = 0.5


def popcount(mask: int) -> int:
    return bin(mask).count('1')


def columns(masks) -> int:
    n = 1
    for m in masks:
        n *= popcount(m)
    return n


# =============================================================================
# SIMULATION  (shared by both objectives)
# =============================================================================
def simulate(probs: np.ndarray, n_sims: int, seed: int = 0) -> np.ndarray:
    """(n_sims, n_matches) of sampled outcomes 0/1/2 from the model."""
    rng = np.random.default_rng(seed)
    n = len(probs)
    out = np.empty((n_sims, n), dtype=np.int8)
    for i in range(n):
        p = np.clip(np.asarray(probs[i], float), 1e-9, None)
        out[:, i] = rng.choice(3, size=n_sims, p=p / p.sum())
    return out


def crowd_hit_fraction(omega: np.ndarray, crowd: np.ndarray,
                       threshold: int) -> np.ndarray:
    """Per simulated week, the fraction of the FIELD reaching the threshold.

    A random crowd coupon gets match i right with probability crowd[i][actual],
    so its number correct is Poisson-binomial over those — computed per week
    because it depends on which results actually landed.
    """
    n_sims, n = omega.shape
    qs = np.take_along_axis(crowd.T[None, :, :].repeat(n_sims, 0),
                            omega[:, None, :], axis=1)[:, 0, :]
    # convolve per week: dist over 0..n successes
    dist = np.zeros((n_sims, n + 1))
    dist[:, 0] = 1.0
    for i in range(n):
        q = qs[:, i:i + 1]
        dist[:, 1:i + 2] = dist[:, 1:i + 2] * (1 - q) + dist[:, 0:i + 1] * q
        dist[:, 0] *= (1 - q[:, 0])
    return dist[:, threshold:].sum(axis=1)


def pot_share(crowd_frac: np.ndarray, field: int) -> np.ndarray:
    """Your slice of the tier pot in each simulated week.

    Winners split it, and you are one of them, so the share is
    1 / (other winners + 1). That +1 matters: it caps the prize at the whole
    pot. Without it the objective chases weeks where the field is wiped out
    and the implied payout runs to infinity, producing a ticket that wins
    almost never.
    """
    return 1.0 / (field * np.asarray(crowd_frac, float) + 1.0)


def evaluate(masks, omega: np.ndarray, threshold: int,
             share: np.ndarray | None):
    """Return (P(hit), expected pot share) for a candidate ticket."""
    hits = np.zeros(len(omega), dtype=np.int16)
    for i, m in enumerate(masks):
        hits += ((m >> omega[:, i]) & 1).astype(np.int16)
    won = hits >= threshold
    p_hit = float(won.mean())
    if share is None:
        return p_hit, p_hit
    return p_hit, float((won * share).mean())


# =============================================================================
# OPTIMISER
# =============================================================================
def build(probs, threshold, budget, crowd=None, n_sims=20000, seed=0,
          restarts=6, field=DEFAULT_FIELD, tilt=DEFAULT_TILT):
    """Choose the symbols for every match subject to columns <= budget.

    Seeded with the top-k-by-probability ticket (optimal for the hit
    objective), then hill-climbed over single-match and pairwise symbol
    changes — which matters for the payout objective, where the best pair is
    not always the two most likely outcomes.
    """
    probs = [np.asarray(p, float) / np.asarray(p, float).sum() for p in probs]
    n = len(probs)
    share = omega = None
    if crowd is not None:
        # payout needs the joint distribution of results, so simulate
        omega = simulate(probs, n_sims, seed)
        crowd = np.asarray(crowd, float)
        crowd = crowd / crowd.sum(axis=1, keepdims=True)
        share = pot_share(crowd_hit_fraction(omega, crowd, threshold), field)
        if tilt < 1.0:
            # tilt=0 ignores the crowd entirely, 1 follows the payout fully
            share = share ** tilt

    def topk_mask(i, k):
        order = np.argsort(-probs[i])[:k]
        m = 0
        for o in order:
            m |= 1 << int(o)
        return m

    def exact_hit(masks):
        """P(>= threshold) in closed form — no sampling noise."""
        qs = [float(sum(probs[i][o] for o in range(3) if (masks[i] >> o) & 1))
              for i in range(n)]
        return toto.prob_at_least(qs, threshold)

    def score(masks):
        # The hit objective has an exact Poisson-binomial answer, so use it:
        # optimising a sampled estimate would chase Monte-Carlo noise and pick
        # a ticket that only looks better. Payout has no closed form.
        if share is None:
            return exact_hit(masks)
        return evaluate(masks, omega, threshold, share)[1]

    # greedy seed: spend the budget on the least predictable matches
    seed_masks = [topk_mask(i, 1) for i in range(n)]
    cols = 1
    while True:
        best = None
        for i in range(n):
            k = popcount(seed_masks[i])
            if k >= 3:
                continue
            new_cols = cols // k * (k + 1)
            if new_cols > budget:
                continue
            trial = list(seed_masks)
            trial[i] = topk_mask(i, k + 1)
            gain = score(trial) - score(seed_masks)
            eff = gain / (new_cols - cols)
            if gain > 0 and (best is None or eff > best[0]):
                best = (eff, i, new_cols, trial)
        if best is None:
            break
        _, _, cols, seed_masks = best

    best_masks, best_score = list(seed_masks), score(seed_masks)
    rng = np.random.default_rng(seed)
    for attempt in range(max(1, restarts)):
        cur = list(seed_masks) if attempt == 0 else _random_start(n, budget, rng)
        cur_s = score(cur)
        improved = True
        while improved:
            improved = False
            for i in range(n):                        # single-match moves
                keep, best_local = cur[i], cur_s
                for m in ALL_MASKS:
                    if m == keep:
                        continue
                    cur[i] = m
                    if columns(cur) <= budget:
                        s = score(cur)
                        if s > best_local + 1e-12:
                            best_local, keep, improved = s, m, True
                    cur[i] = keep
                cur[i] = keep
                cur_s = max(cur_s, best_local)
            for i in range(n):                        # pairwise moves
                for j in range(i + 1, n):
                    ki, kj, best_local = cur[i], cur[j], cur_s
                    bi, bj = ki, kj
                    for mi in ALL_MASKS:
                        for mj in ALL_MASKS:
                            if mi == ki and mj == kj:
                                continue
                            cur[i], cur[j] = mi, mj
                            if columns(cur) <= budget:
                                s = score(cur)
                                if s > best_local + 1e-12:
                                    best_local, bi, bj = s, mi, mj
                    cur[i], cur[j] = bi, bj
                    if best_local > cur_s + 1e-12:
                        cur_s, improved = best_local, True
        if cur_s > best_score + 1e-12:
            best_masks, best_score = list(cur), cur_s

    if share is None:
        p_hit, ev = exact_hit(best_masks), exact_hit(best_masks)
    else:
        p_hit, ev = evaluate(best_masks, omega, threshold, share)
    return {
        'masks': best_masks,
        'labels': [MASK_LABEL[m] for m in best_masks],
        'columns': columns(best_masks),
        'p_hit': p_hit,
        'ev': ev,
        'objective': 'payout' if share is not None else 'hit',
    }


def sweep(probs, threshold, budget, crowd, field=DEFAULT_FIELD,
          n_sims=40000, seed=7, tilts=(0.0, 0.25, 0.5, 0.65, 0.8, 1.0)):
    """Hit-rate vs payout across tilt values, all scored on ONE yardstick.

    Each ticket is built with its own tilt but evaluated against the same
    untilted pot share, otherwise the numbers are not comparable and the
    trade-off is invisible.
    """
    pn = [np.asarray(p, float) / np.asarray(p, float).sum() for p in probs]
    crowd = np.asarray(crowd, float)
    crowd = crowd / crowd.sum(axis=1, keepdims=True)
    omega = simulate(pn, n_sims, seed)
    share = pot_share(crowd_hit_fraction(omega, crowd, threshold), field)
    rows = []
    for t in tilts:
        r = build(probs, threshold, budget, crowd=crowd, tilt=t,
                  n_sims=n_sims // 2, field=field, restarts=3)
        p_hit, ev = evaluate(r['masks'], omega, threshold, share)
        rows.append({'tilt': t, 'p_hit': p_hit, 'ev': ev,
                     'ticket': ' '.join(r['labels']), 'columns': r['columns'],
                     'masks': r['masks']})
    base = rows[0]['ev'] or 1.0
    for r in rows:
        r['vs_base'] = r['ev'] / base
    return rows


def _random_start(n, budget, rng):
    masks = [1 << int(rng.integers(3)) for _ in range(n)]
    cols = 1
    for i in rng.permutation(n):
        for k in (3, 2):
            if cols * k <= budget and rng.random() < 0.4:
                extra = [o for o in range(3) if not (masks[i] >> o) & 1]
                rng.shuffle(extra)
                for o in extra[:k - 1]:
                    masks[i] |= 1 << o
                cols *= k
                break
    return masks


# =============================================================================
# REPORTING
# =============================================================================
def render(rows, result, threshold, budget, probs, crowd=None) -> str:
    out = []
    obj = result['objective']
    out.append(f"\n  YOUR TICKET — {result['columns']} columns "
               f"(budget {budget}), optimised for "
               f"{'expected payout' if obj == 'payout' else 'hit chance'}\n")
    head = f"  {'#':>2}  {'Match':<40} {'Play':<5} {'cols':>4}  {'1':>4} {'X':>4} {'2':>4}"
    if crowd is not None:
        head += f"  {'crowd 1/X/2':>14}  {'edge':>6}"
    out.append(head)
    out.append('  ' + '-' * (len(head) - 2))
    for i, r in enumerate(rows):
        p = probs[i]
        lbl = result['labels'][i]
        k = popcount(result['masks'][i])
        line = (f"  {i+1:>2}  {r['match'][:40]:<40} {lbl:<5} "
                f"{('x' + str(k)) if k > 1 else '':>4}  "
                f"{p[0]:>4.0%} {p[1]:>4.0%} {p[2]:>4.0%}")
        if crowd is not None:
            c = np.asarray(crowd[i], float); c = c / c.sum()
            # how much more we like our pick than the field does
            picked = [o for o in range(3) if (result['masks'][i] >> o) & 1]
            edge = max(p[o] / max(c[o], 1e-6) for o in picked)
            line += (f"  {c[0]:>4.0%}/{c[1]:>3.0%}/{c[2]:>3.0%}  "
                     f"{edge:>5.2f}x")
        out.append(line)
    out.append('')
    out.append(f"  P(>= {threshold} correct) = {result['p_hit']:.2%}")
    if obj == 'payout':
        out.append(f"  Expected share of the prize tier = {result['ev']:.2e} "
                   "of the pot per week (compare between tickets)")
    singles = sum(1 for m in result['masks'] if popcount(m) == 1)
    doubles = sum(1 for m in result['masks'] if popcount(m) == 2)
    triples = sum(1 for m in result['masks'] if popcount(m) == 3)
    out.append(f"  {singles} singles, {doubles} doubles, {triples} triples")
    return '\n'.join(out)


def parse_crowd(text):
    """'68/21/11' or '68 21 11' per line -> array, or None."""
    if not text:
        return None
    rows = []
    for line in str(text).splitlines():
        nums = [x for x in line.replace('/', ' ').replace(',', '.').split()
                if x.replace('.', '').isdigit()]
        if len(nums) >= 3:
            rows.append([float(x) for x in nums[-3:]])
    return np.array(rows) if rows else None


def main():
    ap = argparse.ArgumentParser(description='Build the actual Toto ticket')
    ap.add_argument('--coupon', required=True)
    ap.add_argument('--game', choices=list(toto.GAMES), default='turkish')
    ap.add_argument('--budget', type=int, default=100)
    ap.add_argument('--objective', choices=['hit', 'payout'], default='hit')
    ap.add_argument('--crowd', help='file with "68/21/11" per match')
    ap.add_argument('--sims', type=int, default=20000)
    ap.add_argument('--field', type=int, default=DEFAULT_FIELD,
                    help='how many coupons the field plays (order of '
                         'magnitude is what matters)')
    ap.add_argument('--tilt', type=float, default=DEFAULT_TILT,
                    help=f'0 = ignore the crowd, 1 = full payout tilt '
                         f'(default {DEFAULT_TILT}; above ~0.75 the hit rate '
                         f'collapses)')
    ap.add_argument('--sweep', action='store_true',
                    help='show the hit-vs-payout trade-off instead of one ticket')
    args = ap.parse_args()

    n_exp, threshold, _ = toto.GAMES[args.game]
    df = pd.read_csv(args.coupon)
    df.columns = [c.strip().lower() for c in df.columns]
    for c in ('o1', 'ox', 'o2'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    ctx = toto._load_context()
    rows, probs = [], []
    for _, r in df.iterrows():
        p, src, _ = toto.match_probs(r, ctx)
        if p is None:
            p, src = np.array([1 / 3, 1 / 3, 1 / 3]), 'no data'
        rows.append({'match': f"{r['home']} v {r['away']}", 'src': src})
        probs.append(p)

    crowd = None
    if args.objective == 'payout':
        if args.crowd and os.path.isfile(args.crowd):
            with open(args.crowd, encoding='utf-8') as f:
                crowd = parse_crowd(f.read())
        if crowd is None or len(crowd) != len(probs):
            sys.exit('--objective payout needs --crowd with one "1/X/2" '
                     'percentage line per match')

    if len(df) != n_exp:
        print(f"  ! {len(df)} matches, {args.game} expects {n_exp}")

    if args.sweep:
        if crowd is None:
            sys.exit('--sweep needs --crowd')
        print(f"\n  TRADE-OFF  ({args.budget} columns, field ~{args.field:,})\n")
        print(f"  {'tilt':>5} {'P(>=' + str(threshold) + ')':>9} "
              f"{'payout':>9} {'every':>8}   ticket")
        print('  ' + '-' * 74)
        for r in sweep(probs, threshold, args.budget, crowd, field=args.field):
            every = f"{1 / max(r['p_hit'], 1e-9):.0f} wk"
            print(f"  {r['tilt']:>5} {r['p_hit']:>9.2%} "
                  f"{r['vs_base']:>8.2f}x {every:>8}   {r['ticket'][:34]}")
        print("\n  Payout is relative to tilt 0. Pick the largest tilt whose "
              "hit rate\n  you can still live with — past the cliff it is a "
              "lottery ticket.")
        return
    res = build(probs, threshold, args.budget, crowd=crowd,
                n_sims=args.sims, field=args.field, tilt=args.tilt)
    print(render(rows, res, threshold, args.budget, probs, crowd))


if __name__ == '__main__':
    main()
