import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from batches import partition_data, training_data


class BatchTests(unittest.TestCase):
    def test_native_packing_keeps_all_checkpoints(self):
        rows = [dict(tokens=[1, 2, 0], response_length=1, loss_mask=[1], reward=.7,
                     group_index=g, metadata=dict(lane='critic', node_id=n))
                for g in range(6) for n in range(g+1)]
        data = training_data(rows, lane='critic', expected_groups=range(6))
        args = SimpleNamespace(calculate_per_token_loss=False, global_batch_size=6,
                               use_dynamic_batch_size=True, max_tokens_per_gpu=24,
                               balance_by_flops=False, balance_data=False)
        config = dict(dp_size=2, cp_size=1, vpp_size=1, microbatch_group_size_per_vp_stage=1)
        packets = partition_data(args, config, data)
        self.assertEqual(sum(len(p['tokens']) for p in packets), 21)
        self.assertTrue(all(p['global_batch_sizes'] == [6] for p in packets))

    def test_native_reducer_matches_question_edge_mean(self):
        # Two edges in question 0, one in question 1. The first edge is split
        # across two spans/microbatches. Equal-question weighting survives it.
        rows = []
        losses = []
        for group, node, edge_tokens, token_losses in [(0, 1, 4, [2]), (0, 1, 4, [2,2,2]),
                                                     (0, 2, 2, [6,6]), (1, 1, 1, [10])]:
            n = len(token_losses)
            rows.append(dict(tokens=[0]+[1]*n, response_length=n, loss_mask=[1]*n,
                             reward=1, rollout_log_probs=[-.1]*n, group_index=group,
                             metadata=dict(lane='actor', node_id=node, edge_tokens=edge_tokens)))
            losses.append(torch.tensor(token_losses, dtype=torch.float32))
        data = training_data(rows, lane='actor', expected_groups=[0,1])
        root = Path(__file__).resolve().parents[4]
        path = root / 'slime/slime/backends/megatron_utils/cp_utils.py'
        tree = ast.parse(path.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_sum_of_sample_mean')
        mod = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), fn], type_ignores=[])
        ns = dict(torch=torch, mpu=SimpleNamespace(get_context_parallel_world_size=lambda:1))
        exec(compile(ast.fix_missing_locations(mod), str(path), 'exec'), ns)
        subtotal = 0.
        for i, loss in enumerate(losses):
            reducer = ns['get_sum_of_sample_mean']([len(rows[i]['tokens'])], [len(loss)],
                       [torch.ones_like(loss)], torch.tensor([data['rollout_mask_sums'][i]]))
            subtotal += reducer(loss).item()
        self.assertEqual(subtotal / 2, 7.)  # ((2+6)/2 + 10)/2

    def test_masked_placeholder_preserves_empty_question_in_batch(self):
        rows = [dict(tokens=[1, 2], response_length=1,
                     loss_mask=[0] if group == 5 else [1], reward=0.,
                     rollout_log_probs=[-.1], group_index=group,
                     metadata=dict(lane='actor', node_id=1, edge_tokens=1))
                for group in range(6)]
        data = training_data(rows, lane='actor', expected_groups=range(6))
        self.assertEqual(set(data['rollout_ids']), set(range(6)))
        self.assertEqual(data['loss_masks'][-1], [0])
        self.assertEqual(data['rollout_mask_sums'][-1], 1)


if __name__ == '__main__':
    unittest.main()
