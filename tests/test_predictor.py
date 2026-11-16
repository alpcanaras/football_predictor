"""The predictor's fallbacks: odds alone, national teams, old data, name clashes."""
import os
import tempfile
import unittest

os.environ.setdefault('FOOTBALL_PREDICTOR_TOTO_DIR', tempfile.mkdtemp(prefix='fp_test_toto_'))

import numpy as np  # noqa: E402

from scripts import market, v2_goals  # noqa: E402
from scripts.predictor import Predictor  # noqa: E402

_P = None


def predictor():
    global _P
    if _P is None:
        _P = Predictor()
        _P.bundles()
    return _P


class OddsAlone(unittest.TestCase):
    def test_a_pair_no_model_knows_is_priced_from_its_odds(self):
        odds = [2.10, 3.30, 3.40]
        r = predictor().predict('Nowhere United', 'Elsewhere Rovers', odds=odds)
        self.assertTrue(r['ok'])
        self.assertEqual((r['kind'], r['src'], r['model_1x2']), ('odds', 'odds', None))
        np.testing.assert_allclose(r['p1x2'], market.shin(odds), atol=1e-9)
        self.assertLess(r['fit_error'], 1e-6)
        self.assertEqual(r['ou_src'], 'implied')
        self.assertTrue(0 < r['ou']['3.5'] < r['ou']['2.5'] < r['ou']['1.5'] < 1)
        self.assertTrue(r['notes'])

    def test_its_over_under_odds_are_used_too(self):
        r = predictor().predict('Nowhere United', 'Elsewhere Rovers', odds=[1.30, 5.5, 9.0], ou_odds=[1.55, 2.40])
        self.assertEqual(r['ou_src'], 'odds')
        self.assertAlmostEqual(r['ou']['2.5'], (1 / 1.55) / (1 / 1.55 + 1 / 2.40), places=6)

    def test_without_odds_there_is_nothing_to_say(self):
        r = predictor().predict('Nowhere United', 'Elsewhere Rovers')
        self.assertFalse(r['ok'])
        self.assertFalse(predictor().predict('Nowhere United', 'Elsewhere Rovers', odds=[1.0, 3.0, 4.0])['ok'])

    def test_a_known_club_against_an_unknown_one_also_falls_back_to_the_odds(self):
        r = predictor().predict('Galatasaray', 'Elsewhere Rovers', odds=[1.25, 6.0, 11.0])
        self.assertEqual(r['kind'], 'odds')
        self.assertEqual(r['sides']['home']['canonical'], 'Galatasaray')
        self.assertIsNone(r['sides']['away']['canonical'])


class Nations(unittest.TestCase):
    def test_the_home_side_gets_home_advantage_unless_the_ground_is_neutral(self):
        P = predictor()
        home, flat = P.predict('Norway', 'Italy'), P.predict('Norway', 'Italy', neutral=True)
        self.assertEqual((home['kind'], home['neutral'], flat['neutral']), ('national', False, True))
        self.assertGreater(home['p1x2'][0], flat['p1x2'][0] + .04)
        self.assertLess(home['p1x2'][2], flat['p1x2'][2] - .04)

    def test_a_name_that_is_a_club_and_a_nation_follows_its_opponent(self):
        P = predictor()
        self.assertEqual(P.predict('Andorra', 'Malta')['kind'], 'national')
        self.assertEqual(P.predict('Malta', 'Andorra')['kind'], 'national')
        self.assertEqual(P.side('Andorra')['kind'], 'club')            # alone it stays the club


class OldData(unittest.TestCase):
    def test_a_club_that_left_its_league_is_flagged(self):
        P = predictor()
        league, team, mate = None, None, None
        for (lg, t), last in sorted(P.team_last.items(), key=lambda kv: kv[1]):
            cur = P.current[lg]
            if P.clubs.get(t) == lg and t not in cur['teams'] and (cur['last'] - cur['start']).days > 60:
                league, team, mate = lg, t, sorted(cur['teams'])[0]
                break
        self.assertIsNotNone(team, 'the data has no club that left its league?')
        r = P.predict(mate, team)
        self.assertEqual(r['kind'], 'club')
        self.assertEqual(list(r['old_data']), ['away'])
        self.assertIn('old data', ' '.join(r['notes']))
        self.assertIsNone(P.predict(mate, sorted(P.current[league]['teams'])[1])['old_data'])


class Calibration(unittest.TestCase):
    def test_switching_it_off_gives_the_plain_tilt(self):
        P = predictor()
        keep = P.calib
        try:
            P.calib = None
            plain = P.predict('Galatasaray', 'Fenerbahce', odds=[2.1, 3.4, 3.4])
        finally:
            P.calib = keep
        served = P.predict('Galatasaray', 'Fenerbahce', odds=[2.1, 3.4, 3.4])
        np.testing.assert_allclose(plain['p1x2'], served['p1x2'], atol=1e-9)      # the result never moves
        if keep:
            self.assertNotAlmostEqual(plain['scores'][0]['p'], served['scores'][0]['p'], places=4)


if __name__ == '__main__':
    unittest.main()
