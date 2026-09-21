import subprocess
import unittest
from datetime import datetime
from ppo_leases import refresh


class LeaseTests(unittest.TestCase):
    def test_transient_failure_keeps_original_deadline_and_counts_down(self):
        end = '2026-09-16T09:31:37'
        expiry = datetime.fromisoformat(end).timestamp()
        def fail(*args, **kwargs):
            raise subprocess.CalledProcessError(1, args[0])
        leases, errors = refresh(['1'], [{'job': '1', 'end': end}], now=expiry-5000, query=fail)
        self.assertEqual(leases[0]['remaining_seconds'], 5000)
        self.assertEqual(leases[0]['source'], 'cached')
        self.assertTrue(errors)

    def test_missing_initial_deadline_requests_drain_without_raising(self):
        def timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired(args[0], 10)
        leases, errors = refresh(['1'], [], now=10, query=timeout)
        self.assertEqual(leases[0]['remaining_seconds'], 0)
        self.assertEqual(leases[0]['source'], 'unknown')
        self.assertTrue(errors)

    def test_recovery_and_partial_output(self):
        end = '2026-09-16T09:31:37'
        def query(command, **kwargs):
            self.assertIn('--jobs=1,2', command)
            self.assertEqual(kwargs['timeout'], 10)
            return f'1|{end}\n2|Unknown\n'
        leases, errors = refresh(['1','2'], [{'job':'2','end':end}], now=0, query=query)
        self.assertEqual([r['source'] for r in leases], ['slurm','cached'])
        self.assertEqual(set(errors), {'2'})


if __name__ == '__main__':
    unittest.main()
