"""Behavioral checks for causality, data semantics and artifact cutoffs."""
from argparse import Namespace
import itertools
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import joblib
import numpy as np
import pandas as pd

from scripts.research_features import FeatureState, attach_odds, feature_columns, replay, season_id
from scripts.research_train import fit_bundle, predict_bundle, temporal_split


def match(day, home='A', away='B', hg=1, ag=0, **extra):
    return dict(Date=pd.Timestamp(day), league='british_pl', HomeTeam=home,
                AwayTeam=away, FTHG=hg, FTAG=ag,
                FTR='H' if hg > ag else ('A' if hg < ag else 'D'),
                SeasonId=season_id('british_pl', day), **extra)


class CausalFeatures(unittest.TestCase):
    def test_future_results_cannot_change_earlier_features(self):
        raw = pd.DataFrame([match('2025-08-01'), match('2025-08-08', hg=0, ag=2),
                            match('2025-08-15', hg=3, ag=1)])
        original, _ = replay(raw)
        changed = raw.copy()
        changed.loc[2, ['FTHG', 'FTAG', 'FTR']] = [0, 7, 'A']
        actual, _ = replay(changed)
        pd.testing.assert_frame_equal(original[feature_columns()], actual[feature_columns()])

    def test_same_day_order_does_not_matter(self):
        raw = pd.DataFrame([match('2025-08-01'), match('2025-08-01', 'C', 'D', 0, 2),
                            match('2025-08-08', 'A', 'C')])
        a, _ = replay(raw)
        b, _ = replay(raw.iloc[[1, 0, 2]])
        pd.testing.assert_frame_equal(a, b)
        self.assertTrue(a.iloc[:2].LeagueHomeRate.isna().all())
        self.assertEqual(a.iloc[2].LeagueHomeRate, .5)

    def test_live_features_equal_replay_prefix(self):
        raw = pd.DataFrame([match('2025-08-01', HY=0, HR=0, AY=1, AR=0),
                            match('2025-08-08', HY=2, HR=0, AY=0, AR=0),
                            match('2025-08-15')])
        historical, _ = replay(raw)
        _, state = replay(raw.iloc[:2])
        live = state.fixture(raw.iloc[2])
        np.testing.assert_allclose(historical.iloc[2][feature_columns()].astype(float),
                                   list(live.values()), equal_nan=True)
        self.assertEqual(live['Home_cards_10'], 1.)
        self.assertEqual(live['Home_cards_observed'], 2)

    def test_missing_is_not_zero(self):
        raw = pd.DataFrame([match('2025-08-01', HST=0), match('2025-08-08'), match('2025-08-15')])
        out, _ = replay(raw)
        self.assertEqual(out.iloc[2]['Home_sot_10'], 0.)
        self.assertEqual(out.iloc[2]['Home_sot_observed'], 1)
        self.assertTrue(np.isnan(out.iloc[2]['Away_sot_10']))

    def test_season_boundary_regresses_without_mutating_lookup(self):
        _, state = replay(pd.DataFrame([match('2025-05-01')]))
        fixture = match('2025-07-25')
        rating = state.teams[('british_pl', 'A')].rating
        a, b = state.fixture(fixture), state.fixture(fixture)
        np.testing.assert_allclose(list(a.values()), list(b.values()), equal_nan=True)
        self.assertEqual(a['HomeSeasonN'], 0)
        self.assertEqual(state.teams[('british_pl', 'A')].rating, rating)
        self.assertEqual(season_id('usa', '2025-08-01'), '2025')
        self.assertEqual(season_id('british_pl', '2025-07-25'), '2025/2026')

    def test_state_cannot_predict_its_past(self):
        _, state = replay(pd.DataFrame([match('2025-08-01')]))
        with self.assertRaises(ValueError):
            state.fixture(match('2025-08-01'))
        with self.assertRaises(ValueError):
            state.update_day(pd.DataFrame([match('2025-08-01')]))

    def test_duplicate_team_day_rejected(self):
        with self.assertRaises(ValueError):
            replay(pd.DataFrame([match('2025-08-01'), match('2025-08-01', 'A', 'C')]))


class OddsAndArtifacts(unittest.TestCase):
    def test_closing_prices_never_substitute_for_preclose(self):
        d = pd.DataFrame([dict(AvgCH=2., AvgCD=3., AvgCA=4.)])
        out = attach_odds(d)
        self.assertTrue(out[['qA', 'qD', 'qH']].isna().all().all())

    def test_incomplete_triplet_falls_back_as_whole(self):
        d = pd.DataFrame([dict(AvgH=2., AvgD=np.nan, AvgA=4., B365H=3., B365D=3., B365A=3.)])
        out = attach_odds(d)
        np.testing.assert_allclose(out[['qA', 'qD', 'qH']], [[1/3]*3])
        self.assertEqual(out.iloc[0].odds_source, 'B365H/B365D/B365A')

    def test_split_keeps_whole_dates_and_excludes_future(self):
        frame = pd.DataFrame({'Date': pd.to_datetime(['2025-01-01', '2025-06-01', '2025-06-01',
                                                      '2025-07-01', '2025-08-01'])})
        fit, cal = temporal_split(frame, '2025-07-01', 30, 365)
        self.assertEqual(len(fit), 1)
        self.assertEqual(len(cal), 2)
        self.assertLess(fit.Date.max(), cal.Date.min())
        with self.assertRaises(ValueError):
            temporal_split(frame, '2024-01-01')

    def test_trained_artifact_roundtrip_and_cutoff_enforcement(self):
        rows = [match(day, hg=[0, 1, 2][i % 3], ag=1,
                      AvgH=2.5, AvgD=3.3, AvgA=2.8, B365H=2.4, B365D=3.2, B365A=2.7)
                for i, day in enumerate(pd.date_range('2023-01-01', periods=650))]
        frame, _ = replay(attach_odds(pd.DataFrame(rows)))
        args = Namespace(calibration_days=90, lookback_days=700, trees=4,
                         half_life=500, elo_k=20, season_retention=.85)
        cutoff = frame.Date.iloc[-20]
        bundle = fit_bundle(frame, cutoff, args)
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'bundle.joblib'
            joblib.dump(bundle, path)
            loaded = joblib.load(path)
        out = predict_bundle(loaded, frame[frame.Date >= cutoff])
        for p in out.values():
            self.assertTrue(np.isfinite(p).all())
            np.testing.assert_allclose(p.sum(axis=1), 1., atol=1e-6)
        with self.assertRaises(ValueError):
            predict_bundle(loaded, frame.iloc[:5])
        self.assertLess(pd.Timestamp(bundle['fit_end']), pd.Timestamp(bundle['calibration_start']))
        self.assertLess(pd.Timestamp(bundle['calibration_end']), cutoff)


if __name__ == '__main__':
    unittest.main()
