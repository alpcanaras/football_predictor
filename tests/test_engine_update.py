"""The automatic data update must also fire on a server that stays up."""
import os
import tempfile
import time
import unittest

os.environ.setdefault('FOOTBALL_PREDICTOR_TOTO_DIR', tempfile.mkdtemp(prefix='fp_test_toto_'))

from webapp import engine as engine_mod  # noqa: E402


class KeepFresh(unittest.TestCase):
    def setUp(self):
        self.e = engine_mod.Engine()
        self.started = []
        self.e.start_job = lambda kind: self.started.append(kind) or {'ok': True}
        self.settings = {'auto_update': True, 'last_update': time.time() - 13 * 3600}
        self.e.settings = lambda: dict(self.settings)

    def test_old_data_starts_an_update_while_the_page_is_open(self):
        self.e.state = 'ready'
        self.e.keep_fresh()
        self.assertEqual(self.started, ['all'])

    def test_not_before_the_engine_is_ready_and_not_when_switched_off(self):
        self.e.state = 'loading'
        self.e.keep_fresh()
        self.e.state = 'ready'
        self.settings['auto_update'] = False
        self.e.keep_fresh()
        self.assertEqual(self.started, [])

    def test_fresh_data_is_left_alone(self):
        self.e.state = 'ready'
        self.settings['last_update'] = time.time() - 3600
        self.e.keep_fresh()
        self.assertEqual(self.started, [])

    def test_a_failed_update_is_not_started_again_on_the_next_ping(self):
        self.e.state = 'ready'
        self.e.keep_fresh()                 # starts; it "fails": last_update stays old
        self.e.keep_fresh()
        self.e.keep_fresh()
        self.assertEqual(self.started, ['all'])
        self.e._auto_wait = time.time() - 1  # the wait is over
        self.e.keep_fresh()
        self.assertEqual(self.started, ['all', 'all'])


if __name__ == '__main__':
    unittest.main()
