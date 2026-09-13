import unittest
from allocation_plan import build_plan


class AllocationTests(unittest.TestCase):
    def allocations(self, sizes):
        return {j:dict(host=f'host{j}',gpus=n,expires=1000+j) for j,n in sizes.items()}

    def test_long_lived_eight_gpu_layout(self):
        p = build_plan([1,2],[3,4],4,self.allocations({1:2,2:2,3:2,4:2}))
        self.assertEqual((p['train_nodes'],p['train_gpus_per_node'],p['rollout_gpus'],p['total_gpus']), (2,2,3,8))

    def test_single_host_training_layouts(self):
        for inference, expected in [([2],6),([2,3],8),([2,3,4],10)]:
            with self.subTest(inference=inference):
                p = build_plan([1],inference,inference[-1],self.allocations({1:4,2:2,3:2,4:2}))
                self.assertEqual(p['total_gpus'],expected)
                self.assertEqual(p['train_gpus_per_node'],4)

    def test_bad_allocations_rejected(self):
        a = self.allocations({1:2,2:2,3:2,4:2})
        for train, inference, critic in [([1],[2],2),([1,2],[2,3],3),([1,2],[3,4],1)]:
            with self.subTest(train=train,inference=inference,critic=critic), self.assertRaises(ValueError):
                build_plan(train,inference,critic,a)
        a[2]['host'] = a[1]['host']
        with self.assertRaises(ValueError):
            build_plan([1,2],[3,4],4,a)
