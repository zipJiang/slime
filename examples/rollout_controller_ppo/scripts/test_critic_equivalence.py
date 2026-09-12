import unittest

from critic_equivalence import compare_scores


class CriticEquivalenceTests(unittest.TestCase):
    def check(self, native, replica, repeated=None, **kwargs):
        def result(scores):
            return dict(version='critic-0078', scores=scores)
        return compare_scores(result(native), result(replica),
                              result(replica if repeated is None else repeated),
                              version='critic-0078', count=len(native), **kwargs)

    def test_short_context_agreement_does_not_hide_real_context_failure(self):
        report = self.check([.3106943965, .3174262643, .5860491395],
                            [.3098584116, .3182732165, .5765419602])
        self.assertFalse(report['passed'])
        self.assertEqual(report['failed_context_indices'], [2])
        self.assertEqual(report['worst_context_index'], 2)

    def test_small_deterministic_differences_pass(self):
        self.assertTrue(self.check([.2, .8], [.201, .799])['passed'])

    def test_repeated_inference_must_match(self):
        self.assertFalse(self.check([.2], [.2], [.20001])['passed'])

    def test_invalid_probabilities_fail_closed(self):
        for v in [float('nan'), float('inf'), -.1, 1.1]:
            for side in ['native', 'replica', 'repeated']:
                values = dict(native=[.2], replica=[.2], repeated=[.2])
                values[side] = [v]
                with self.subTest(side=side, value=v), self.assertRaises(ValueError):
                    self.check(**values)

    def test_counts_and_versions_are_checked(self):
        good = dict(version='critic-0078', scores=[.2])
        for bad in [dict(version='critic-0077', scores=[.2]),
                    dict(version='critic-0078', scores=[])]:
            with self.assertRaises(ValueError):
                compare_scores(good, bad, good, version='critic-0078', count=1)


if __name__ == '__main__':
    unittest.main()
