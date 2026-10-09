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


if __name__ == '__main__':
    unittest.main()
