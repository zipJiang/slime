"""Regression checks for sampling with replacement and split isolation."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from balanced_data import (EXPERIMENT, inventories, metadata_for, read_rows,
                           validate_batch, validate_selection, request_seed)


class BalancedDataTests(unittest.TestCase):
    def test_frozen_schedule_preserves_every_cell(self):
        rows = read_rows(EXPERIMENT / 'data/train-schedule.jsonl')
        self.assertEqual(len(rows), 780)
        keys = [r['metadata']['case_key'] for r in rows]
        for i in range(0, len(keys), 6):
            validate_batch(keys[i:i+6])
        self.assertTrue(any(k != metadata_for(k)['family'] for k in keys))

    def test_duplicate_draws_are_separate_valid_roots(self):
        key = next(iter(inventories()['train']))
        with TemporaryDirectory() as tmp:
            questions = Path(tmp)/'questions.json'
            questions.write_text(json.dumps([key, key]))
            self.assertEqual(validate_selection([key, key], 'train', False), 'train')

    def test_test_case_rejected_for_training(self):
        key = next(iter(inventories()['test']))
        with TemporaryDirectory() as tmp:
            questions = Path(tmp)/'questions.json'
            questions.write_text(json.dumps([key]))
            with self.assertRaises(ValueError):
                validate_selection([key], 'train', False)
            with self.assertRaises(ValueError):
                validate_selection([key], 'val', False)

    def test_repeated_roots_have_distinct_reproducible_seeds(self):
        first = request_seed('run/0000', 0, 'case', None, 0)
        self.assertEqual(first, request_seed('run/0000', 0, 'case', None, 0))
        self.assertNotEqual(first, request_seed('run/0000', 1, 'case', None, 0))

    def test_cell_imbalance_is_rejected(self):
        key = next(iter(inventories()['train']))
        with self.assertRaises(ValueError):
            validate_batch([key]*6)


if __name__ == '__main__':
    unittest.main()
