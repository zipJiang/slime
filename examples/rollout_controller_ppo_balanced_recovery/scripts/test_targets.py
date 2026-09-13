"""CPU contract tests, including the actual local Slime causal value slicing."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from targets import checkpoint_fields, prepared_advantages, split_targets


class Tokenizer:
    def encode(self, text, add_special_tokens):
        assert not add_special_tokens
        return list(range(1, len(text) + 1))


def slime_response_extractor():
    # Execute the actual pure slicing function without requiring Megatron/CUDA
    # imports on the login host. Production uses CP=1, TP=2, DP=2.
    root = Path(__file__).resolve().parents[1]
    path = root / 'snapshots/slime/slime/backends/megatron_utils/loss.py'
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_responses')
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), fn], type_ignores=[])
    namespace = dict(torch=torch, mpu=SimpleNamespace(get_context_parallel_world_size=lambda: 1))
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace['get_responses']


class TargetTests(unittest.TestCase):
    def critic(self, context='abc'):
        return dict(context=context, target=.75, group_index=0,
                    metadata=dict(lane='critic', node_id=0, value_version='v1'))

    def test_value_pooling_and_gradient_boundary(self):
        rows = [checkpoint_fields(self.critic(s), Tokenizer(), sentinel_token_id=0,
                                  max_sequence_length=100) for s in ['abc', 'defgh']]
        sizes = [len(r['tokens']) for r in rows]
        logits = torch.arange(sum(sizes), dtype=torch.float32).reshape(1, -1, 1).requires_grad_()
        chunks = list(slime_response_extractor()(logits, args=SimpleNamespace(rollout_temperature=1),
                      unconcat_tokens=[torch.tensor(r['tokens']) for r in rows],
                      total_lengths=sizes, response_lengths=[1, 1], apply_temperature=False))
        values = torch.cat([x[0].flatten() for x in chunks])
        self.assertEqual(values.tolist(), [2., 8.])
        values.sum().backward()
        self.assertEqual(logits.grad.flatten().nonzero().flatten().tolist(), [2, 8])

    def test_no_silent_truncation(self):
        with self.assertRaises(ValueError):
            checkpoint_fields(self.critic(), Tokenizer(), sentinel_token_id=0, max_sequence_length=3)

    def test_warmup_uses_observed_return_without_self_distillation(self):
        record = self.critic()
        record['metadata']['diagnostics'] = dict(observations=4, mean_return=.25)
        row = checkpoint_fields(record, Tokenizer(), sentinel_token_id=0,
                                max_sequence_length=100, warmup=True)
        self.assertEqual(row['reward'], .25)
        self.assertEqual(record['target'], .75)
        self.assertEqual(row['metadata']['target_source'], 'warmup_empirical_suffix_mean')

    def test_prepared_values_are_not_centered_or_recomputed(self):
        data = dict(rewards=[.25, -.75], rollout_log_probs=[torch.zeros(3), torch.zeros(2)])
        prepared_advantages(SimpleNamespace(normalize_advantages=False), data)
        self.assertEqual([v.tolist() for v in data['advantages']], [[.25]*3, [-.75]*2])
        data['returns'][0][0] = 9
        self.assertEqual(data['advantages'][0][0].item(), .25)

    def test_span_partition_invariance_and_lane_isolation(self):
        rows=[]
        for i, n in enumerate([1, 3]):
            rows.append(dict(tokens=[9]+[2]*n, loss_mask=[0]+[1]*n,
                             logprobs=[0]+[-.4]*n, reward=.25, group_index=0,
                             metadata=dict(lane='actor', node_id=1, estimator='refined_td',
                                           span_count=2, span_index=i, edge_tokens=4)))
        rows.append(self.critic())
        actors, critics = split_targets(rows)
        self.assertEqual([s['metadata']['edge_loss_scale'] for s in actors], [.25,.75])
        self.assertNotIn('tokens', critics[0])
        for invalid in [rows[1:], rows+[rows[-1]]]:
            with self.assertRaises(ValueError):
                split_targets(invalid)


if __name__ == '__main__':
    unittest.main()
