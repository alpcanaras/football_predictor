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


class LiveSeason(unittest.TestCase):
    """A live fixture has no season of its own. It must land in the season the
    league's newest matches use, whatever notation the provider writes."""

    def state(self, league, season, days):
        st = FeatureState()
        for k, day in enumerate(days):
            h, a = ('A', 'B') if k % 2 == 0 else ('B', 'A')
            st.update_day(pd.DataFrame([dict(Date=pd.Timestamp(day), league=league, HomeTeam=h,
                                             AwayTeam=a, FTHG=1, FTAG=0, FTR='H', SeasonId=season)]))
        return st

    def played(self, st, league, day):
        f = st.fixture(dict(Date=pd.Timestamp(day), league=league, HomeTeam='A', AwayTeam='B'))
        return f['HomeSeasonN']

    def test_calendar_league_that_moved_to_a_split_season(self):
        # 'japan' is on the calendar-year list, but its file says 2026/2027
        st = self.state('japan', '2026/2027', ['2026-08-08', '2026-08-15', '2026-11-08'])
        self.assertEqual(self.played(st, 'japan', '2026-11-21'), 3)     # same season
        self.assertEqual(self.played(st, 'japan', '2027-02-20'), 3)     # ... into spring
        self.assertEqual(self.played(st, 'japan', '2027-08-07'), 0)     # the next one

    def test_calendar_and_split_notations_as_before(self):
        st = self.state('usa', '2026', ['2026-03-01', '2026-10-17'])
        self.assertEqual(self.played(st, 'usa', '2026-11-01'), 2)
        self.assertEqual(self.played(st, 'usa', '2027-02-27'), 0)
        st = self.state('german', '2026/2027', ['2026-08-22', '2026-11-07'])
        self.assertEqual(self.played(st, 'german', '2026-11-21'), 2)
        self.assertEqual(self.played(st, 'german', '2027-08-21'), 0)

    def test_a_replayed_match_keeps_the_season_from_its_file(self):
        st = self.state('japan', '2026/2027', ['2026-08-08'])
        f = st.fixture(dict(Date=pd.Timestamp('2026-08-15'), league='japan', HomeTeam='A',
                            AwayTeam='B', SeasonId='2026'))
        self.assertEqual(f['HomeSeasonN'], 0)


if __name__ == '__main__':
    unittest.main()
