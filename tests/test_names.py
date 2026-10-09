"""The team-name matcher: full names, aliases, typos — and refusing to guess."""
import unittest

import pandas as pd

from scripts import names, toto

_IDX = None


def index():
    global _IDX
    if _IDX is None:
        d = pd.read_csv('full_processed_data.csv', usecols=['league', 'HomeTeam', 'AwayTeam'],
                        low_memory=False)
        clubs = {}
        for lg, h, a in zip(d.league, d.HomeTeam, d.AwayTeam):
            clubs.setdefault(h, lg)
            clubs.setdefault(a, lg)
        _IDX = names.NameIndex(clubs, sorted(toto._load_intl()['ratings']))
    return _IDX


class Resolve(unittest.TestCase):
    def check(self, typed, canonical, how=None):
        m = index().resolve(typed)
        self.assertIsNotNone(m, f'{typed!r} should resolve')
        self.assertEqual(m.canonical, canonical, f'{typed!r} -> {m}')
        if how:
            self.assertEqual(m.how, how)

    def test_the_examples_that_started_this(self):
        self.check('Basaksehir', 'Buyuksehyr')
        self.check('Başakşehir', 'Buyuksehyr')
        self.check('Istanbul Basaksehir FK', 'Buyuksehyr')
        self.check('VfL Wolfsburg', 'Wolfsburg')
        self.check('buyuksehr', 'Buyuksehyr', 'fuzzy')

    def test_full_names_and_nicknames(self):
        for typed, canon in [('Bayern München', 'Bayern Munich'), ('FC Bayern', 'Bayern Munich'),
                             ('Manchester United', 'Man United'), ('Man Utd', 'Man United'),
                             ('Newcastle United', 'Newcastle'), ('Borussia Mönchengladbach', "M'gladbach"),
                             ('1. FC Köln', 'FC Koln'), ('Eintracht Frankfurt', 'Ein Frankfurt'),
                             ('PSG', 'Paris SG'), ('Inter Milan', 'Inter'), ('AC Milan', 'Milan'),
                             ('Atletico Madrid', 'Ath Madrid'), ('Athletic Bilbao', 'Ath Bilbao'),
                             ('Bayer 04 Leverkusen', 'Leverkusen'), ('Beşiktaş JK', 'Besiktas'),
                             ('Çaykur Rizespor', 'Rizespor'), ('Sporting CP', 'Sp Lisbon')]:
            self.check(typed, canon)

    def test_nations(self):
        self.check('Türkiye', 'Turkey')
        self.check('Czechia', 'Czech Republic')
        self.check('Holland', 'Netherlands')
        self.check('USA', 'United States')

    def test_a_typo_is_flagged_as_a_guess(self):
        m = index().resolve('Fenerbache')
        self.assertEqual((m.canonical, m.how, m.sure), ('Fenerbahce', 'fuzzy', False))

    def test_it_refuses_to_guess(self):
        for typed in ['Paris', 'Real', 'United', 'Union', 'Atletico', 'Zzzz Nonexistent']:
            self.assertIsNone(index().resolve(typed), typed)

    def test_suggestions_for_what_it_will_not_guess(self):
        got = [index().display_of[c] for c, _ in index().suggest('Real', 6)]
        self.assertIn('Real Madrid', got)

    def test_display_names(self):
        self.assertEqual(index().display_of['Buyuksehyr'], 'İstanbul Başakşehir')
        self.assertEqual(index().display_of['Wolfsburg'], 'VfL Wolfsburg')

    def test_remembered_pick_wins(self):
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            old = os.environ.get('FOOTBALL_PREDICTOR_TOTO_DIR')
            os.environ['FOOTBALL_PREDICTOR_TOTO_DIR'] = tmp
            try:
                idx = names.NameIndex({'Galatasaray': 'turkish', 'Fenerbahce': 'turkish'})
                self.assertIsNone(idx.resolve('the yellow canaries', fuzzy=False))
                self.assertTrue(idx.remember('the yellow canaries', 'Fenerbahce'))
                self.assertEqual(idx.resolve('The Yellow Canaries').canonical, 'Fenerbahce')
            finally:
                if old is None:
                    os.environ.pop('FOOTBALL_PREDICTOR_TOTO_DIR', None)
                else:
                    os.environ['FOOTBALL_PREDICTOR_TOTO_DIR'] = old


class Tables(unittest.TestCase):
    def test_every_curated_name_is_a_real_name_in_the_data(self):
        idx = index()
        self.assertEqual([k for k in names.TEAMS if k not in idx.clubs], [])
        nations = set(idx.nations)
        self.assertEqual([k for k in names.NATIONS if k not in nations], [])

    def test_no_alias_points_at_two_teams(self):
        idx = index()
        clash = {k: v for k, v in idx._alias.items() if len(v) > 1}
        self.assertEqual(clash, {}, 'an alias must name one team')


if __name__ == '__main__':
    unittest.main()
