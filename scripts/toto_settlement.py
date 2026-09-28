"""Exact full-system tier counts and explicit money-based scenario evaluation.

This is independent of the legacy hit/payout optimizer. A ticket is a Cartesian
product of selections, using ticket.py's bit masks (home=1, draw=2, away=4).
Payouts are per winning column in each exact tier, not per covered match or
per winning system. Monetary inputs must use a common currency.
"""
from __future__ import annotations

import numpy as np


def _masks(masks):
    m = np.asarray(masks)
    if m.ndim != 1 or not 1 <= len(m) <= 30 or not np.isin(m, np.arange(1, 8)).all():
        raise ValueError('Need 1..30 match masks, each an integer from 1 to 7')
    return m.astype(np.int64)


def tier_counts(masks, outcomes):
    """Return (scenarios, n+1) integer counts of columns with exactly r hits."""
    masks = _masks(masks)
    outcomes = np.asarray(outcomes)
    if outcomes.ndim == 1:
        outcomes = outcomes[None, :]
    if outcomes.ndim != 2 or outcomes.shape[1] != len(masks) or not np.isin(outcomes, [0, 1, 2]).all():
        raise ValueError('Outcomes must be 0=home, 1=draw, 2=away for every match')
    outcomes = outcomes.astype(np.int64)
    n = len(masks)
    count = np.zeros((len(outcomes), n+1), dtype=np.int64)
    count[:, 0] = 1
    for i, mask in enumerate(masks):
        k = int(mask).bit_count()
        covered = ((mask >> outcomes[:, i]) & 1)[:, None]
        previous = count[:, :i+1].copy()
        count[:, :i+2] = 0
        count[:, :i+1] += previous * (k-covered)
        count[:, 1:i+2] += previous * covered
    return count


def _money(values, n):
    if not values or any(not isinstance(k, (int, np.integer)) or k < 0 or k > n for k in values):
        raise ValueError('Supply exact prize tiers within the ticket length')
    if any(not np.isfinite(v) or v < 0 for v in values.values()):
        raise ValueError('Money amounts must be finite and nonnegative')
    return values


def _cost(masks, column_cost, fee):
    if not np.isfinite([column_cost, fee]).all() or column_cost < 0 or fee < 0:
        raise ValueError('Costs must be finite and nonnegative')
    columns = int(np.prod([int(m).bit_count() for m in masks]))
    return columns, columns*column_cost+fee


def settle(masks, actual, payouts_per_column, column_cost, fee=0.):
    """Settle one completed system against published per-column tier payouts."""
    masks = _masks(masks)
    count = tier_counts(masks, actual)
    if len(count) != 1:
        raise ValueError('Settlement needs exactly one realized outcome vector')
    payouts = _money(payouts_per_column, len(masks))
    columns, cost = _cost(masks, column_cost, fee)
    gross = sum(count[0, tier]*payout for tier, payout in payouts.items())
    return dict(columns=columns, winning_columns={int(t): int(count[0, t]) for t in payouts},
                cost=float(cost), gross=float(gross), net=float(gross-cost))


def scenarios(probs, crowd, other_columns, n=20000, seed=42):
    """Sample results and competing winner counts for an explicit crowd model.

    Assumptions: independent match outcomes; independent public columns with
    selections independently drawn from supplied match marginals. Real public
    systems can violate these assumptions. Field size counts OTHER columns,
    not players. Reuse these scenarios when comparing candidate systems; use
    another seed for final evaluation of a selected candidate.
    """
    p, q = np.asarray(probs, float), np.asarray(crowd, float)
    if p.ndim != 2 or p.shape[1] != 3 or p.shape != q.shape or not 1 <= len(p) <= 30:
        raise ValueError('Probability and crowd arrays must both have shape (matches, 3)')
    if (not np.isfinite(p).all() or not np.isfinite(q).all() or (p < 0).any()
            or (q < 0).any() or (p.sum(axis=1) <= 0).any() or (q.sum(axis=1) <= 0).any()):
        raise ValueError('Invalid probability or crowd distribution')
    if not isinstance(other_columns, (int, np.integer)) or other_columns < 0 or not isinstance(n, int) or n < 2:
        raise ValueError('Need nonnegative integer field size and >=2 simulations')
    p, q = p/p.sum(axis=1, keepdims=True), q/q.sum(axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    actual = np.column_stack([rng.choice(3, n, p=row) for row in p])
    distribution = np.zeros((n, len(p)+1))
    distribution[:, 0] = 1
    for i in range(len(p)):
        hit = q[i, actual[:, i], None]
        previous = distribution[:, :i+1].copy()
        distribution[:, :i+2] = 0
        distribution[:, :i+1] += previous*(1-hit)
        distribution[:, 1:i+2] += previous*hit
    distribution /= distribution.sum(axis=1, keepdims=True)
    competitors = np.array([rng.multinomial(other_columns, row) for row in distribution])
    return actual, competitors


def evaluate(masks, actual, competitors, tier_pots, column_cost, fee=0.):
    """Expected net money under the supplied scenarios and tier pots.

    Own winning columns appear in numerator AND denominator. Multiple tiers
    can pay on the same system. Rollover effects must already be represented
    in the supplied pots. This does not impose a jurisdiction's rollover rules.
    """
    masks = _masks(masks)
    counts = tier_counts(masks, actual)
    rivals = np.asarray(competitors)
    if rivals.shape != counts.shape or not np.isfinite(rivals).all() or (rivals < 0).any() or (rivals != np.floor(rivals)).any():
        raise ValueError('Need nonnegative integer competing winner counts by scenario and tier')
    pots = _money(tier_pots, len(masks))
    columns, cost = _cost(masks, column_cost, fee)
    gross = np.zeros(len(counts))
    for tier, pot in pots.items():
        total = counts[:, tier] + rivals[:, tier]
        share = np.divide(counts[:, tier], total, out=np.zeros(len(counts)), where=total > 0)
        gross += pot*share
    net = gross-cost
    return dict(columns=columns, cost=float(cost), expected_gross=float(gross.mean()),
                expected_net=float(net.mean()), probability_any_prize=float((counts[:, list(pots)].sum(axis=1) > 0).mean()),
                probability_profit=float((net > 0).mean()),
                monte_carlo_standard_error=float(net.std(ddof=1)/np.sqrt(len(net))) if len(net) > 1 else None)
