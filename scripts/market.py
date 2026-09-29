"""
Bookmaker odds -> probabilities
===============================
One place for removing the bookmaker margin, used by the coupon, the match /
fixtures anchoring and the blend fitting alike.

Shin's method, not proportional normalisation. Bookmakers load more margin
onto longshots (the favourite-longshot bias), so dividing 1/odds by their sum
overprices underdogs. Shin models that directly — a fraction z of the market
is treated as better-informed money — and solves for the z that makes the
probabilities sum to one.

Measured on 8,920 matches out of sample (2025/26 on, pre-match average odds,
since that is what the live feed carries):

    proportional            1.00419
    proportional ^ 1.11     1.00349   (the old anchor path's fitted fudge)
    shin                    1.00337   (t = +3.2 vs proportional)

Shin beats the tuned temperature with no fitted parameter at all; the
temperature Shin itself would want is ~1.03, i.e. it already corrects what
the 1.11 was crudely approximating. On 24,398 matches from 2023 the gain over
proportional is t = +4.3.
"""

from __future__ import annotations

import numpy as np


def shin(odds) -> np.ndarray:
    """Margin-free probabilities from decimal odds.

    `odds` is (n_outcomes,) or (n_rows, n_outcomes); any column order — the
    output keeps it. Rows with missing or non-positive odds come back NaN.
    Vectorised bisection on z, so it is cheap on whole datasets.
    """
    o = np.asarray(odds, dtype=float)
    single = o.ndim == 1
    o = np.atleast_2d(o)
    out = np.full(o.shape, np.nan)
    ok = np.all(np.isfinite(o) & (o > 1.0), axis=1)
    if not ok.any():
        return out[0] if single else out

    pi = 1.0 / o[ok]
    S = pi.sum(axis=1, keepdims=True)

    def probs(z):
        z = z[:, None]
        return (np.sqrt(z ** 2 + 4.0 * (1.0 - z) * pi ** 2 / S) - z) \
            / (2.0 * (1.0 - z))

    # sum(probs(z)) falls as z rises; no margin (S <= 1) means z = 0
    lo = np.zeros(len(pi))
    hi = np.full(len(pi), 0.5)
    for _ in range(60):
        mid = (lo + hi) / 2.0
        too_big = probs(mid).sum(axis=1) > 1.0
        lo = np.where(too_big, mid, lo)
        hi = np.where(too_big, hi, mid)
    z = np.where(S[:, 0] > 1.0, (lo + hi) / 2.0, 0.0)
    p = probs(z)
    out[ok] = p / p.sum(axis=1, keepdims=True)
    return out[0] if single else out


def proportional(odds) -> np.ndarray:
    """Plain 1/odds normalisation. Kept for comparison and O/U 2.5, whose
    blend weights were fitted against it."""
    o = np.atleast_2d(np.asarray(odds, dtype=float))
    inv = 1.0 / o
    out = inv / inv.sum(axis=1, keepdims=True)
    return out[0] if np.asarray(odds).ndim == 1 else out
