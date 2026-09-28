import itertools
import unittest

import numpy as np

from scripts.toto_settlement import evaluate, scenarios, settle, tier_counts


class SystemSettlement(unittest.TestCase):
    def test_tier_counts_match_enumeration(self):
        # Every result for varied systems, including draws/away-only selections.
        outcomes = np.array(list(itertools.product(range(3), repeat=4)))
        for masks in ([1, 2, 4, 7], [3, 5, 6, 7], [7, 7, 7, 7]):
            columns = list(itertools.product(*[[o for o in range(3) if m & (1 << o)] for m in masks]))
            expected = np.zeros((len(outcomes), 5), dtype=int)
            for i, actual in enumerate(outcomes):
                for col in columns:
                    expected[i, sum(a == b for a, b in zip(actual, col))] += 1
            np.testing.assert_array_equal(tier_counts(masks, outcomes), expected)

    def test_multiple_winning_columns_and_tiers_pay(self):
        result = settle([3, 3], [0, 0], {1: 10., 2: 100.}, column_cost=.5)
        self.assertEqual(result['winning_columns'], {1: 2, 2: 1})
        self.assertEqual(result['columns'], 4)
        self.assertEqual(result['gross'], 120.)
        self.assertEqual(result['net'], 118.)

    def test_own_columns_dilute_each_tier_share(self):
        result = evaluate([3, 3], [[0, 0]], [[10, 8, 4]], {1: 100., 2: 200.}, column_cost=.5)
        self.assertEqual(result['expected_gross'], 60.)
        self.assertEqual(result['expected_net'], 58.)

    def test_unwon_pot_does_not_pay_and_fees_count(self):
        result = evaluate([1, 1], [[2, 2]], [[0, 0, 0]], {2: 1000.}, column_cost=.5, fee=1.)
        self.assertEqual(result['expected_gross'], 0.)
        self.assertEqual(result['expected_net'], -1.5)
        self.assertEqual(result['probability_any_prize'], 0.)

    def test_full_turkish_system_preserves_every_column(self):
        counts = tier_counts([7]*15, [0]*15)[0]
        self.assertEqual(counts.sum(), 3**15)
        self.assertEqual(counts[15], 1)
        self.assertEqual(counts[14], 30)
        self.assertEqual(counts[13], 420)
        self.assertEqual(counts[12], 3640)

    def test_simulated_competitors_preserve_field_size(self):
        p = [[.5, .3, .2], [.2, .3, .5]]
        actual, rivals = scenarios(p, p, other_columns=200, n=500)
        self.assertEqual(actual.shape, (500, 2))
        np.testing.assert_array_equal(rivals.sum(axis=1), np.full(500, 200))
        same_actual, same_rivals = scenarios(p, p, other_columns=200, n=500)
        np.testing.assert_array_equal(actual, same_actual)
        np.testing.assert_array_equal(rivals, same_rivals)

    def test_deterministic_public_results(self):
        actual, rivals = scenarios([[1, 0, 0]]*2, [[1, 0, 0]]*2, 10, n=10)
        self.assertTrue((actual == 0).all())
        np.testing.assert_array_equal(rivals, np.tile([0, 0, 10], (10, 1)))

    def test_invalid_inputs_fail(self):
        for masks in ([0], [8], [], [1.5]):
            with self.assertRaises(ValueError):
                tier_counts(masks, [0])
        with self.assertRaises(ValueError):
            settle([1], [0], {2: 10}, 1.)
        with self.assertRaises(ValueError):
            settle([1], [0], {1: 10}, -1.)


if __name__ == '__main__':
    unittest.main()
