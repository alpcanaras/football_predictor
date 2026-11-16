"""The scoreline grid tilt: margins must match the targets exactly."""
import unittest

import numpy as np

from scripts import v2_goals as g


class TiltGrid(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.grid = g.grid(rng.uniform(.6, 2.4, 8), rng.uniform(.5, 2.0, 8), -0.04)
        # targets taken from another coherent goal model, so that they can all
        # be true at once (independent random targets can contradict each other)
        other = g.grid_markets(g.grid(rng.uniform(.6, 2.4, 8), rng.uniform(.5, 2.0, 8), -0.04))
        self.p1 = np.c_[other['pH'], other['pD'], other['pA']]
        self.ou = other['ou25']
        self.bt = other['btts']

    def test_margins_match_all_targets(self):
        t = g.tilt_grid(self.grid, self.p1, self.ou, self.bt)
        m = g.grid_markets(t)
        np.testing.assert_allclose(t.sum((1, 2)), 1, atol=1e-9)
        np.testing.assert_allclose(np.c_[m['pH'], m['pD'], m['pA']], self.p1, atol=1e-6)
        np.testing.assert_allclose(m['ou25'], self.ou, atol=1e-6)
        np.testing.assert_allclose(m['btts'], self.bt, atol=1e-6)

    def test_contradictory_targets_still_give_a_valid_grid(self):
        rng = np.random.default_rng(5)
        t = g.tilt_grid(self.grid, rng.dirichlet([1, 1, 1], 8), rng.uniform(.05, .95, 8), rng.uniform(.05, .95, 8))
        self.assertTrue(np.isfinite(t).all())
        np.testing.assert_allclose(t.sum((1, 2)), 1, atol=1e-9)

    def test_no_targets_changes_nothing(self):
        np.testing.assert_allclose(g.tilt_grid(self.grid), self.grid)

    def test_a_single_target_is_hit_without_moving_the_rest_of_the_shape(self):
        t = g.tilt_grid(self.grid, p1x2=self.p1)
        m = g.grid_markets(t)
        np.testing.assert_allclose(np.c_[m['pH'], m['pD'], m['pA']], self.p1, atol=1e-9)
        # cells inside one result keep their relative weights
        home = (g._I > g._J)
        ratio = t[:, home] / self.grid[:, home]
        np.testing.assert_allclose(ratio, ratio[:, :1] * np.ones_like(ratio), rtol=1e-6)


class ServedGrid(unittest.TestCase):
    """The grid the app serves: odds grid, mix, shape multipliers, then the tilt."""

    def setUp(self):
        rng = np.random.default_rng(3)
        self.model = g.grid(rng.uniform(.7, 2.3, 6), rng.uniform(.6, 1.9, 6), -0.05)
        other = g.grid_markets(g.grid(rng.uniform(.7, 2.3, 6), rng.uniform(.6, 1.9, 6), -0.03))
        self.p1 = np.c_[other['pH'], other['pD'], other['pA']]
        self.ou = other['ou25']
        self.calib = {'enabled': True, 'weight_model': .5, 'rho': -0.04,
                      'theta': {k: [0, -.13, -.12, 0, -.09, -.06, 0, .10, .23]
                                for k in ('model', 'odds', 'odds_ou', 'nomodel', 'nomodel_ou')}}

    def margins(self, t):
        m = g.grid_markets(t)
        return np.c_[m['pH'], m['pD'], m['pA']], m['ou25']

    def test_rates_reproduce_the_1x2_they_were_read_from(self):
        rates = g.rates_from_1x2(self.p1)
        back = g.grid_markets(g.grid(rates[:, 0], rates[:, 1], g.ODDS_RHO))
        np.testing.assert_allclose(np.c_[back['pH'], back['pD'], back['pA']], self.p1, atol=1e-6)
        self.assertTrue(np.isnan(g.rates_from_1x2([[.5, np.nan, .2]])).all())

    def test_without_a_calibration_it_is_the_plain_tilt(self):
        np.testing.assert_allclose(g.serve(self.model, self.p1, self.ou), g.tilt_grid(self.model, self.p1, self.ou))
        off = dict(self.calib, enabled=False)
        np.testing.assert_allclose(g.serve(self.model, self.p1, self.ou, g.rates_from_1x2(self.p1), off),
                                   g.tilt_grid(self.model, self.p1, self.ou))

    def test_every_mode_lands_on_its_targets(self):
        rates = g.rates_from_1x2(self.p1)
        for model, ou, r in ((self.model, self.ou, rates), (self.model, None, rates), (self.model, None, None),
                             (None, self.ou, rates), (None, None, rates)):
            t = g.serve(model, self.p1, ou, r, self.calib)
            p, o = self.margins(t)
            np.testing.assert_allclose(t.sum((1, 2)), 1, atol=1e-9)
            np.testing.assert_allclose(p, self.p1, atol=1e-6)
            if ou is not None:
                np.testing.assert_allclose(o, ou, atol=1e-6)

    def test_the_shape_multipliers_move_mass_to_one_goal_wins(self):
        plain = g.tilt_grid(self.model, self.p1)
        shaped = g.serve(self.model, self.p1, None, None, self.calib)
        one, wide = g.SHAPE_GROUPS['H+1'], g.SHAPE_GROUPS['H+3']
        self.assertTrue((shaped[:, one].sum(1) > plain[:, one].sum(1)).all())
        self.assertTrue((shaped[:, wide].sum(1) < plain[:, wide].sum(1)).all())

    def test_no_model_and_no_odds_is_an_error(self):
        with self.assertRaises(ValueError):
            g.serve(None, self.p1)

    def test_fitted_multipliers_match_the_shares_inside_each_result(self):
        rng = np.random.default_rng(9)
        n = 3000
        seed = g.grid(rng.uniform(.7, 2.3, n), rng.uniform(.6, 1.9, n), -0.04)
        m = g.grid_markets(seed)
        p1 = np.c_[m['pH'], m['pD'], m['pA']]
        truth = g.tilt_grid(seed * g.shape_matrix([0, -.2, -.3, 0, -.1, -.2, 0, .1, .3])[None], p1)
        cum = truth.reshape(n, -1).cumsum(1)
        cell = (rng.random((n, 1)) > cum).sum(1).clip(max=truth[0].size - 1)
        hg, ag = cell // (g.G + 1), cell % (g.G + 1)
        theta = g.fit_shape(seed, p1, None, hg, ag)
        fitted = g.tilt_grid(seed * g.shape_matrix(theta)[None], p1)
        # the tilt fixes how much each result gets; the multipliers decide how
        # that is shared out (of the home wins, how many are by one goal, ...)
        masks = list(g.SHAPE_GROUPS.values())
        for a in range(0, 9, 3):
            got = np.array([fitted[:, m].sum() for m in masks[a:a + 3]])
            want = np.array([m[hg, ag].sum() for m in masks[a:a + 3]], float)
            np.testing.assert_allclose(got / got.sum(), want / want.sum(), atol=1e-4)
        self.assertEqual((theta[0], theta[3], theta[6]), (0, 0, 0))


if __name__ == '__main__':
    unittest.main()
