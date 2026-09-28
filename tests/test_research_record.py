import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.research_record import forecast


class ProspectiveRecording(unittest.TestCase):
    def test_only_future_fixtures_and_prior_results_are_used(self):
        bundle = dict(cutoff='2026-09-01', leagues=['british_pl'], elo_k=20, season_retention=.85)
        raw = pd.DataFrame([
            dict(Date=pd.Timestamp(day), league='british_pl', HomeTeam='A', AwayTeam='B',
                 FTHG=hg, FTAG=0, FTR='H')
            for day, hg in [('2026-09-01', 1), ('2026-09-27', 9), ('2026-09-29', 8)]])
        fx = pd.DataFrame([dict(Date=pd.Timestamp(day), league='british_pl', HomeTeam='A', AwayTeam='B',
                                feed='rich', odds_source='AvgH/AvgD/AvgA', betA=3., betD=3., betH=3.)
                           for day in ['2026-09-26', '2026-09-27', '2026-09-28']])
        def predict(_, frame):
            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.iloc[0].HomeHistoryN, 1)
            self.assertEqual(frame.iloc[0].Home_gf_10, 1.)
            return {'football': np.array([[.2, .3, .5]])}
        with patch('scripts.research_record.predict_bundle', side_effect=predict):
            result = forecast(bundle, raw, fx, '2026-09-27')
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['Date'], '2026-09-28')
        self.assertEqual(result[0]['state_last_completed_day'], '2026-09-01')

    def test_model_from_future_is_rejected(self):
        with self.assertRaises(ValueError):
            forecast({'cutoff': '2026-10-01'}, pd.DataFrame(), pd.DataFrame(), '2026-09-27')


if __name__ == '__main__':
    unittest.main()
