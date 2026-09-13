import copy
import unittest
from advantage_diagnostics import summarize


class AdvantageDiagnosticsTests(unittest.TestCase):
    def test_spans_share_an_edge_and_small_advantages_are_retained(self):
        rows = [dict(group_index=0,metadata=dict(node_id=node),reward=advantage,loss_mask=mask)
                for node, advantage, mask in [(1,.001,[1,0,1]),(1,.001,[1]),(2,0.,[1]),(3,-.5,[1,1])]]
        before=copy.deepcopy(rows)
        report=summarize(rows)
        self.assertEqual(rows,before)
        self.assertEqual(report['edges'],3)
        self.assertEqual(report['tokens'],6)
        self.assertAlmostEqual(report['abs_le_0_001_token_fraction'],4/6)
        self.assertAlmostEqual(report['abs_le_0_0_edge_fraction'],1/3)
