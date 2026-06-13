"""
Dixon-Coles Baseline Model
===========================
Classic Dixon-Coles (1997) bivariate-Poisson-with-correction model, fitted
per league with exponential time decay. Produces 1X2 / O-U / BTTS
probabilities from team attack/defense rates, used as a blend component
alongside the GBM ensemble and bookmaker odds.

Parameters per league: attack_i, defense_i per team (sum-to-zero), a global
goal mean, home advantage, and the low-score correction rho.
"""

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize

DC_HALF_LIFE_DAYS = 390.0      # exponential down-weighting of old matches
DC_WINDOW_DAYS = 1100          # ignore matches older than ~3 years entirely
MAX_GOALS = 10


def _tau(x, y, lam, mu, rho):
    """Dixon-Coles low-score adjustment factor."""
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    if x == 0 and y == 1:
        return 1.0 + lam * rho
    if x == 1 and y == 0:
        return 1.0 + mu * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


class DixonColesModel:
    def __init__(self, half_life_days: float = DC_HALF_LIFE_DAYS):
        self.half_life_days = half_life_days
        self.teams_: list[str] = []
        self.attack_: dict[str, float] = {}
        self.defense_: dict[str, float] = {}
        self.mu_ = 0.0          # log mean goals
        self.home_adv_ = 0.25
        self.rho_ = 0.0

    # ------------------------------------------------------------------
    def fit(self, matches: pd.DataFrame, as_of=None):
        """Fit on matches with columns Date/HomeTeam/AwayTeam/FTHG/FTAG.

        Only matches strictly before `as_of` are used (leak-free by
        construction). Returns self, or None if there is too little data.
        """
        m = matches.dropna(subset=['FTHG', 'FTAG']).copy()
        if as_of is not None:
            as_of = pd.Timestamp(as_of)
            m = m[m['Date'] < as_of]
        else:
            as_of = m['Date'].max()
        m = m[m['Date'] >= as_of - pd.Timedelta(days=DC_WINDOW_DAYS)]
        if len(m) < 80:
            return None

        teams = sorted(set(m['HomeTeam']) | set(m['AwayTeam']))
        t_idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        hi = m['HomeTeam'].map(t_idx).to_numpy()
        ai = m['AwayTeam'].map(t_idx).to_numpy()
        hg = m['FTHG'].to_numpy(int)
        ag = m['FTAG'].to_numpy(int)
        age_days = (as_of - m['Date']).dt.days.to_numpy(float)
        w = np.exp(-math.log(2.0) * age_days / self.half_life_days)

        log_fact_h = np.array([math.lgamma(g + 1) for g in hg])
        log_fact_a = np.array([math.lgamma(g + 1) for g in ag])

        low_mask = (hg <= 1) & (ag <= 1)

        def nll(params):
            att = params[:n]
            dfn = params[n:2 * n]
            mu, home_adv, rho = params[2 * n], params[2 * n + 1], params[2 * n + 2]
            att = att - att.mean()
            dfn = dfn - dfn.mean()

            log_lam = mu + home_adv + att[hi] - dfn[ai]
            log_mu_ = mu + att[ai] - dfn[hi]
            lam = np.exp(np.clip(log_lam, -7, 3))
            mu_a = np.exp(np.clip(log_mu_, -7, 3))

            ll = (hg * log_lam - lam - log_fact_h
                  + ag * log_mu_ - mu_a - log_fact_a)

            # tau correction only touches 0/1-goal scores
            tau = np.ones(len(hg))
            lm = low_mask
            t00 = (hg == 0) & (ag == 0)
            t01 = (hg == 0) & (ag == 1)
            t10 = (hg == 1) & (ag == 0)
            t11 = (hg == 1) & (ag == 1)
            tau[t00] = 1.0 - lam[t00] * mu_a[t00] * rho
            tau[t01] = 1.0 + lam[t01] * rho
            tau[t10] = 1.0 + mu_a[t10] * rho
            tau[t11] = 1.0 - rho
            tau = np.clip(tau, 1e-10, None)
            ll = ll + np.where(lm, np.log(tau), 0.0)

            return -(w * ll).sum() / w.sum()

        x0 = np.zeros(2 * n + 3)
        x0[2 * n] = math.log(max(hg.mean(), 0.5))
        x0[2 * n + 1] = 0.25
        bounds = ([(-2.5, 2.5)] * (2 * n)
                  + [(-1.0, 1.5), (-0.2, 0.8), (-0.3, 0.3)])
        res = minimize(nll, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 300})

        att = res.x[:n] - res.x[:n].mean()
        dfn = res.x[n:2 * n] - res.x[n:2 * n].mean()
        self.teams_ = teams
        self.attack_ = dict(zip(teams, att))
        self.defense_ = dict(zip(teams, dfn))
        self.mu_ = res.x[2 * n]
        self.home_adv_ = res.x[2 * n + 1]
        self.rho_ = res.x[2 * n + 2]
        return self

    # ------------------------------------------------------------------
    def knows(self, team: str) -> bool:
        return team in self.attack_

    def lambdas(self, home: str, away: str):
        lam = math.exp(self.mu_ + self.home_adv_
                       + self.attack_[home] - self.defense_[away])
        mu = math.exp(self.mu_ + self.attack_[away] - self.defense_[home])
        return lam, mu

    def score_grid(self, home: str, away: str) -> np.ndarray:
        lam, mu = self.lambdas(home, away)
        g = np.arange(MAX_GOALS + 1)
        ph = np.exp(-lam) * lam ** g / np.array([math.factorial(int(i)) for i in g])
        pa = np.exp(-mu) * mu ** g / np.array([math.factorial(int(i)) for i in g])
        grid = np.outer(ph, pa)
        for x in (0, 1):
            for y in (0, 1):
                grid[x, y] *= max(_tau(x, y, lam, mu, self.rho_), 1e-10)
        return grid / grid.sum()

    def market_probs(self, home: str, away: str) -> dict:
        grid = self.score_grid(home, away)
        i, j = np.indices(grid.shape)
        total = i + j
        return {
            'p_home': grid[i > j].sum(),
            'p_draw': grid[i == j].sum(),
            'p_away': grid[i < j].sum(),
            'p_over15': grid[total > 1.5].sum(),
            'p_over25': grid[total > 2.5].sum(),
            'p_over35': grid[total > 3.5].sum(),
            'p_btts': grid[(i > 0) & (j > 0)].sum(),
        }
