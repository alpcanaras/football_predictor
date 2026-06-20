"""
Toto Coupon Optimizer
=====================
For pools games (Turkish Spor Toto: 15 matches, prize at 12+;
German Toto 13er Wette: 13 matches, prize at 10+). You predict 1X2 for the
whole coupon and win a tier when enough columns are correct. Your opponent is
the CROWD, not a sharp bookmaker, so the job is:

  1. Get each match's calibrated 1X2 probabilities (bookmaker odds, refined by
     our models where the teams are covered).
  2. Compute your probability of reaching the prize threshold.
  3. If you play a SYSTEM (cover uncertain matches with doubles/triples), spend
     a fixed column budget where it buys the most threshold-probability.

The system maths (exact): submitting a system expands to one column per
combination. The BEST column gets match i right iff the true outcome is among
the outcomes you covered for it. So per match:
    single  -> P(correct) = top-1 prob,           cost x1 columns
    double  -> P(correct) = top-2 probs summed,    cost x2 columns
    triple  -> P(correct) = 1.0 (guaranteed),      cost x3 columns
and the number correct in the best column is a sum of independent Bernoullis
(Poisson-binomial). We maximize P(correct >= threshold) under columns <= budget.

Coupon CSV columns (header row; extra columns ignored):
    home, away              (required)
    o1, ox, o2              (optional bookmaker 1X2 odds, e.g. 2.10/3.30/3.40)
    league                  (optional league key to help model lookup)

    python scripts/toto.py --coupon mycoupon.csv --game turkish --budget 64
    python scripts/toto.py --coupon mycoupon.csv --game german  --budget 48
    python scripts/toto.py --template            # write a blank coupon.csv
"""

import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

GAMES = {                       # (n_matches, prize_threshold, top_tier)
    'turkish': (15, 12, 15),
    'german': (13, 10, 13),
}
OUTCOMES = ['1', 'X', '2']      # home / draw / away


def _fold(s):
    """Lowercase + strip accents, so 'curacao' matches 'Curaçao', 'besiktas'
    matches 'Beşiktaş', 'koln' matches 'Köln', etc."""
    import unicodedata
    s = unicodedata.normalize('NFKD', str(s))
    return ''.join(c for c in s if not unicodedata.combining(c)).lower().strip()


# =============================================================================
# PROBABILITIES PER MATCH
# =============================================================================
def _devig(o1, ox, o2):
    inv = np.array([1.0 / o1, 1.0 / ox, 1.0 / o2])
    return inv / inv.sum()


def _load_blend_weights():
    import json
    from scripts import config
    try:
        with open(os.path.join(config.MODELS_DIR, 'blend_weights.json'),
                  encoding='utf-8') as f:
            w = json.load(f)['with_odds']
        return float(w['model']), float(w['book'])
    except Exception:
        return 0.0, 1.0


def match_probs(row, ctx):
    """Return (probs[1,X,2], source, model_probs_or_None).

    Priority: blend(model, odds) when both exist; else whichever is present.
    The model layer tries the club ensemble first, then falls back to the
    national-team (international/World Cup) Elo+Poisson model, so coupons full
    of national sides (WC weeks) get real probabilities, not 1/3-1/3-1/3.
    `ctx` holds the loaded club-model context (or None to skip the club lookup).
    """
    has_odds = all(pd.notna(row.get(c)) for c in ('o1', 'ox', 'o2'))
    book = _devig(row['o1'], row['ox'], row['o2']) if has_odds else None

    model, kind = None, None
    if ctx is not None:
        model = _model_probs(row, ctx)        # club ensemble [1,X,2] or None
        if model is not None:
            kind = 'model'
    if model is None:                          # national-team fallback
        intl = _intl_probs(row.get('home'), row.get('away'))
        if intl is not None:
            model, kind = intl, 'intl'

    if book is not None and model is not None:
        w_m, w_b = ctx['weights'] if ctx else (0.0, 1.0)
        logp = w_m * np.log(np.clip(model, 1e-9, 1)) + w_b * np.log(book)
        p = np.exp(logp - logp.max()); p /= p.sum()
        return p, 'blend', model
    if book is not None:
        return book, 'odds', model
    if model is not None:
        return model, kind, model
    return None, 'none', None


def _model_probs(row, ctx):
    """Look up the home team's league, build features, predict 1X2 -> [1,X,2]."""
    from scripts import utils
    home = _fuzzy(row['home'], ctx['teams'])
    away = _fuzzy(row['away'], ctx['teams'])
    if not home or not away:
        return None
    try:
        from scripts import predict as predict_mod
        pred = predict_mod.predict_match(
            home, away, ctx['team_stats'], ctx['team_to_league'],
            ctx['hist'], include_xg=False)
        p = pred.get('1x2')
        if not p:
            return None
        return np.array([p['home'], p['draw'], p['away']])
    except Exception:
        return None


def _fuzzy(name, teams):
    if name in teams:
        return name
    low = _fold(name)
    hits = [t for t in teams if _fold(t) == low]
    if hits:
        return hits[0]
    hits = [t for t in teams if low and (low in _fold(t) or _fold(t) in low)]
    return hits[0] if len(hits) == 1 else None


# --- national-team (World Cup / international) fallback --------------------
_INTL_CACHE = None
_INTL_ALIASES = {                       # common shorthands -> dataset name
    'usa': 'United States', 'united states of america': 'United States',
    'uae': 'United Arab Emirates', 'korea': 'South Korea',
    'czechia': 'Czech Republic', 'bosnia': 'Bosnia and Herzegovina',
    'drc': 'DR Congo', "cote d'ivoire": 'Ivory Coast', 'turkiye': 'Turkey',
    'cape verde islands': 'Cape Verde', 'holland': 'Netherlands',
    'ivory coast': 'Ivory Coast',
}


def _load_intl():
    """Load + cache international Elo ratings and the Poisson goal model once."""
    global _INTL_CACHE
    if _INTL_CACHE is not None:
        return _INTL_CACHE
    try:
        from scripts import international as intl
        df = intl.load_results()
        ratings, hist = intl.run_elo(df)
        model = intl.GoalModel().fit(hist)
        _INTL_CACHE = {'ratings': ratings, 'model': model,
                       'names': {_fold(t): t for t in ratings}}
    except Exception:
        _INTL_CACHE = {'ratings': {}, 'model': None, 'names': {}}
    return _INTL_CACHE


def _intl_name(name, ic):
    low = _fold(name)
    low = _fold(_INTL_ALIASES.get(low, low))
    if low in ic['names']:
        return ic['names'][low]
    hits = [orig for l, orig in ic['names'].items()
            if low and (low in l or l in low)]
    return hits[0] if len(hits) == 1 else None


def _intl_probs(home, away, neutral=True):
    """National-team 1X2 via the international model (neutral venue default)."""
    if home is None or away is None:
        return None
    ic = _load_intl()
    if ic['model'] is None:
        return None
    h, a = _intl_name(home, ic), _intl_name(away, ic)
    if not h or not a:
        return None
    p = ic['model'].market_probs(ic['ratings'][h], ic['ratings'][a],
                                 neutral=neutral)
    return np.array([p['p_home'], p['p_draw'], p['p_away']])


def _load_context():
    try:
        from scripts import data_loader, utils
        hist = data_loader.load_processed_data()
        return {
            'hist': hist,
            'team_stats': utils.get_team_stats_table(hist),
            'team_to_league': utils.get_team_to_league_map(hist),
            'teams': set(utils.get_all_teams(hist)),
            'weights': _load_blend_weights(),
        }
    except Exception as e:
        print(f"  (model context unavailable, odds-only: {e})")
        return None


# =============================================================================
# POISSON-BINOMIAL + SYSTEM OPTIMIZER
# =============================================================================
def pb_distribution(qs):
    """Distribution of the count of successes for independent Bernoulli(qs)."""
    dist = np.array([1.0])
    for q in qs:
        dist = np.convolve(dist, [1.0 - q, q])
    return dist


def prob_at_least(qs, t):
    d = pb_distribution(qs)
    return float(d[t:].sum()) if t < len(d) else 0.0


def covered_prob(sorted_probs, c):
    """Sum of the top-c outcome probabilities for one match."""
    return float(np.sum(sorted_probs[:c]))


def optimize_system(per_match_sorted, threshold, budget):
    """Greedy coverage allocation. per_match_sorted: list of descending [p1,p2,p3].

    Returns (coverage list in {1,2,3}, columns_used, P(>=threshold)).
    """
    n = len(per_match_sorted)
    cov = [1] * n
    cols = 1

    def qs():
        return [covered_prob(per_match_sorted[i], cov[i]) for i in range(n)]

    cur_p = prob_at_least(qs(), threshold)
    while True:
        best = None
        for i in range(n):
            if cov[i] >= 3:
                continue
            new_cols = cols // cov[i] * (cov[i] + 1)
            if new_cols > budget:
                continue
            cov[i] += 1
            gain = prob_at_least(qs(), threshold) - cur_p
            cov[i] -= 1
            # marginal gain per extra column spent
            eff = gain / (new_cols - cols) if new_cols > cols else 0.0
            if gain > 1e-6 and (best is None or eff > best[0]):
                best = (eff, i, new_cols, gain)
        if best is None:
            break
        _, i, new_cols, gain = best
        cov[i] += 1
        cols = new_cols
        cur_p += gain
    return cov, cols, cur_p


# =============================================================================
# DRIVER
# =============================================================================
def run(coupon_path, game, budget, use_model=True):
    n_exp, threshold, top_tier = GAMES[game]
    df = pd.read_csv(coupon_path)
    df.columns = [c.strip().lower() for c in df.columns]
    for c in ('o1', 'ox', 'o2'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    if len(df) != n_exp:
        print(f"  ! coupon has {len(df)} matches, {game} expects {n_exp} "
              f"(continuing anyway)")

    ctx = _load_context() if use_model else None

    print(f"\n  {game.upper()} TOTO  —  {len(df)} matches, prize at "
          f"{threshold}+, top tier {top_tier}\n")
    print(f"  {'#':>2} {'Match':<40} {'1':>5} {'X':>5} {'2':>5}  "
          f"{'Pick':>4} {'Src':<6} flag")
    print('  ' + '-' * 78)

    sorted_probs, picks, top1 = [], [], []
    for i, (_, r) in enumerate(df.iterrows(), 1):
        p, src, model = match_probs(r, ctx)
        if p is None:
            print(f"  {i:>2} {r['home']+' v '+r['away']:<40}  "
                  f"NO ODDS / NO MODEL — add o1/ox/o2")
            sorted_probs.append(np.array([0.34, 0.33, 0.33]))
            picks.append('1'); top1.append(0.34)
            continue
        order = np.argsort(-p)
        pick = OUTCOMES[order[0]]
        # contrarian flag: model's pick differs from the odds favourite
        flag = ''
        if model is not None and src == 'blend':
            if np.argmax(model) != np.argmax(_devig(r['o1'], r['ox'], r['o2'])):
                flag = '⚠ model contrarian'
        label = f"{r['home']} v {r['away']}"
        print(f"  {i:>2} {label:<40} {p[0]:>5.0%} {p[1]:>5.0%} {p[2]:>5.0%}  "
              f"{pick:>4} {src:<6} {flag}")
        sorted_probs.append(np.sort(p)[::-1])
        picks.append(pick); top1.append(float(p[order[0]]))

    # --- single-column baseline ---
    q_single = [sp[0] for sp in sorted_probs]
    exp_correct = sum(q_single)
    print(f"\n  Single column (best pick each): expected correct "
          f"≈ {exp_correct:.1f}/{len(df)}")
    d = pb_distribution(q_single)
    for t in range(top_tier, threshold - 1, -1):
        if t < len(d):
            print(f"    P(>= {t:>2} correct) = {d[t:].sum():6.2%}")

    # --- system optimization ---
    if budget > 1:
        cov, cols, p_thr = optimize_system(sorted_probs, threshold, budget)
        n_dbl = sum(1 for c in cov if c == 2)
        n_trp = sum(1 for c in cov if c == 3)
        print(f"\n  Best system within {budget} columns: uses {cols} columns "
              f"({n_dbl} doubles, {n_trp} triples)")
        print(f"    P(>= {threshold} correct) rises to {p_thr:6.2%} "
              f"(from {d[threshold:].sum():.2%} single)")
        up = [(i + 1, df.iloc[i]['home'], df.iloc[i]['away'], cov[i])
              for i in range(len(cov)) if cov[i] > 1]
        if up:
            print("    Cover these (least predictable) with "
                  "doubles/triples:")
            for idx, h, a, c in up:
                kind = 'TRIPLE (1X2)' if c == 3 else 'double'
                print(f"      #{idx:>2} {h} v {a}  -> {kind}")


def parse_lines(text):
    """Parse a pasted coupon into a DataFrame.

    One match per line: 'Home - Away' (also accepts ' v ', ' vs ', ' x ', ':',
    or a bare '-'). Optional bookmaker odds as the last three numbers on the
    line, e.g. 'Norway - Italy 2.10 3.30 3.40'. Blank/garbage lines are skipped.
    """
    seps = [' - ', ' – ', ' — ', ' vs. ', ' vs ', ' v ', ' x ', ' : ',
            ' V ', ' X ', '–', '—', ' - ', '-', ':']
    recs = []
    for raw in str(text).splitlines():
        line = raw.strip()
        if not line:
            continue
        o1 = ox = o2 = None
        toks = line.split()
        if len(toks) >= 3:                       # trailing 3 numbers -> odds
            try:
                tail = [float(t.replace(',', '.')) for t in toks[-3:]]
                if all(v > 1.0 for v in tail):
                    o1, ox, o2 = tail
                    line = ' '.join(toks[:-3]).strip()
            except ValueError:
                pass
        home, away = line, ''
        for s in seps:
            if s in line:
                home, away = line.split(s, 1)
                break
        if home.strip():
            recs.append({'home': home.strip(), 'away': away.strip(),
                         'o1': o1, 'ox': ox, 'o2': o2})
    return pd.DataFrame(recs, columns=['home', 'away', 'o1', 'ox', 'o2'])


def write_template(path='coupon.csv'):
    pd.DataFrame({
        'home': ['Galatasaray', 'Bayern Munich'],
        'away': ['Fenerbahce', 'Dortmund'],
        'o1': [1.80, 1.50], 'ox': [3.60, 4.50], 'o2': [4.20, 5.50],
        'league': ['turkish', 'german'],
    }).to_csv(path, index=False)
    print(f"  Wrote template -> {path}\n  Fill in 15 (Turkish) or 13 "
          f"(German) rows. Odds optional but recommended.")


def main():
    ap = argparse.ArgumentParser(description='Toto coupon optimizer')
    ap.add_argument('--coupon', help='Path to coupon CSV')
    ap.add_argument('--game', choices=list(GAMES), default='turkish')
    ap.add_argument('--budget', type=int, default=1,
                    help='Max columns for system play (1 = single column)')
    ap.add_argument('--no-model', action='store_true',
                    help='Use odds only, skip model lookup')
    ap.add_argument('--template', action='store_true',
                    help='Write a blank coupon.csv and exit')
    args = ap.parse_args()

    if args.template:
        write_template()
        return
    if not args.coupon:
        ap.error('provide --coupon PATH (or --template to start one)')
    run(args.coupon, args.game, args.budget, use_model=not args.no_model)


if __name__ == '__main__':
    main()
