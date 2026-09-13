import copy
import unittest
from benchmark_compare import compare


class BenchmarkTests(unittest.TestCase):
    def arms(self):
        arguments = dict(lr=1e-6, ppo_critic_lr=5e-6, ppo_pass_tokens=140000,
            ppo_search_concurrency=4, ppo_prior_strength=1, rollout_batch_size=6,
            ppo_seed_namespace='paired', max_tokens_per_gpu=24576, seq_length=32768,
            eps_clip=.2, kl_loss_coef=.01)
        base = dict(execution='sync', resume={'iteration':5}, total_gpus=10,
            recipe_arguments=arguments, tokens_per_second=100, interior_tokens_per_second=100, output_tokens=8000000,
            rows=[dict(round=i, families=['a','b'], pass_tokens=140000,
                       seed_namespace=f'paired/{i}') for i in (6,7,8,9)])
        candidate = copy.deepcopy(base)
        candidate.update(execution='overlap', tokens_per_second=110, interior_tokens_per_second=130)
        return base, candidate

    def test_requires_real_speedup_and_comparable_budget(self):
        a,b = self.arms()
        self.assertTrue(compare(a,b)['prioritize_overlap'])
        b['interior_tokens_per_second'] = 115
        self.assertFalse(compare(a,b)['prioritize_overlap'])
        b['interior_tokens_per_second'] = 140
        b['output_tokens'] *= 1.3
        self.assertFalse(compare(a,b)['prioritize_overlap'])

    def test_steady_gain_cannot_hide_net_regression(self):
        a,b = self.arms()
        b['tokens_per_second'] = 95
        self.assertFalse(compare(a,b)['prioritize_overlap'])

    def test_rejects_changed_questions_or_optimizer_or_gpu_budget(self):
        for mutation in (lambda b: b.update(total_gpus=11),
                         lambda b: b['recipe_arguments'].update(lr=2e-6),
                         lambda b: b['rows'][1].update(families=['foreign'])):
            a,b = self.arms()
            mutation(b)
            with self.assertRaises(ValueError):
                compare(a,b)


if __name__ == '__main__':
    unittest.main()
