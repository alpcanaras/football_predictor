"""build_state (what live prediction uses) must equal the full replay's state."""
import unittest

import numpy as np
import pandas as pd

from scripts.research_features import FeatureState, build_state, replay, season_id


def match(day, home, away, hg, ag, close=(2.0, 3.4, 3.8), over=(1.9, 1.9)):
    return dict(Date=pd.Timestamp(day), league='british_pl', HomeTeam=home, AwayTeam=away,
                FTHG=hg, FTAG=ag, FTR='H' if hg > ag else ('A' if hg < ag else 'D'),
                SeasonId=season_id('british_pl', day), AvgCH=close[0], AvgCD=close[1],
                AvgCA=close[2], **{'AvgC>2.5': over[0], 'AvgC<2.5': over[1]})


class BuildState(unittest.TestCase):
    def test_build_state_equals_replay_state(self):
        raw = pd.DataFrame([match('2025-08-01', 'A', 'B', 1, 0), match('2025-08-01', 'C', 'D', 0, 2),
                            match('2025-08-08', 'B', 'C', 2, 2), match('2025-08-15', 'A', 'D', 1, 0),
                            match('2025-08-22', 'D', 'B', 1, 3)])
        for flags in (dict(), dict(market=True), dict(market=True, totals=True)):
            _, a = replay(raw, FeatureState(**flags))
            b = build_state(raw, FeatureState(**flags))
            nxt = dict(Date=pd.Timestamp('2025-09-01'), league='british_pl', HomeTeam='A', AwayTeam='B')
            fa, fb = a.fixture(nxt), b.fixture(nxt)
            self.assertEqual(list(fa), list(fb))
            np.testing.assert_allclose(list(fa.values()), list(fb.values()), equal_nan=True)


if __name__ == '__main__':
    unittest.main()
