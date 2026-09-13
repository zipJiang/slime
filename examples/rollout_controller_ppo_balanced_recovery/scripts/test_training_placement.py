import unittest
from placement import training_bundle_order


class TrainingPlacementTests(unittest.TestCase):
    def test_shuffled_two_host_ranks_keep_tp_pairs_local(self):
        physical = [('b','1'), ('a','1'), ('b','0'), ('a','0')]
        self.assertEqual(training_bundle_order(physical,4), [3,1,2,0])

    def test_single_four_gpu_host(self):
        self.assertEqual(training_bundle_order([('a',str(i)) for i in [3,1,0,2]],4), [2,1,3,0])

    def test_three_plus_one_cannot_form_local_pairs(self):
        with self.assertRaises(ValueError):
            training_bundle_order([('a','0'),('a','1'),('a','2'),('b','0')],4)

    def test_duplicate_gpu_rejected(self):
        with self.assertRaises(ValueError):
            training_bundle_order([('a','0'),('a',0),('b','0'),('b','1')],4)
