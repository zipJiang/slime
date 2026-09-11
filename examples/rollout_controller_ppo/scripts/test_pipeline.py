"""Prefetch must preserve one-update lag and an exact recovery cursor."""
import unittest
from pipeline import BatchStamp, may_prefetch, save_boundary
from critic_replica import FrozenCriticReplica


class PipelineTests(unittest.TestCase):
    def test_staleness_and_warmup_boundary(self):
        self.assertEqual(BatchStamp(6, 6, 6).lineage(6, overlap=True)['actor_lag'], 0)
        self.assertEqual(BatchStamp(7, 6, 6).lineage(7, overlap=True)['actor_lag'], 1)
        for collected, behavior, learner, overlap in [(8,6,8,True), (6,7,6,True),
                (7,6,7,False), (6,6,7,True)]:
            with self.assertRaises(ValueError):
                BatchStamp(collected, behavior, 6).lineage(learner, overlap=overlap)

    def test_checkpoints_never_include_an_untrained_prefetch(self):
        # Simulate the whole schedule and check both optimizer/version lag and
        # dataset cursor at every warmup, periodic and final save boundary.
        ready_behavior, cursor = None, 0
        for round_id in range(126):
            behavior = round_id if ready_behavior is None else ready_behavior
            if ready_behavior is None:
                cursor += 1
            BatchStamp(round_id, behavior, 6).lineage(round_id, overlap=True)
            save = save_boundary(round_id, 6, 12, 125)
            prefetch = may_prefetch(round_id, warmup_rounds=6, last_round=125,
                                   save_now=save, eval_now=save, stop_requested=False)
            ready_behavior = round_id if prefetch else None
            cursor += int(prefetch)
            if save:
                self.assertEqual(cursor, round_id+1)
                self.assertIsNone(ready_behavior)
        self.assertEqual(cursor, 126)

    def test_stop_and_evaluation_drain(self):
        for save, evaluation, stop in [(True,False,False), (False,True,False), (False,False,True)]:
            self.assertFalse(may_prefetch(10, warmup_rounds=6, last_round=125,
                save_now=save, eval_now=evaluation, stop_requested=stop))

    def test_replica_rejects_publication_and_wrong_versions_during_collection(self):
        replica = object.__new__(FrozenCriticReplica)
        replica.version, replica.active, replica.model = 'critic-6', False, object()
        self.assertEqual(replica.begin('critic-6'), 'critic-6')
        with self.assertRaises(ValueError):
            replica.begin('critic-6')
        with self.assertRaises(ValueError):
            replica.end('critic-7')
        self.assertEqual(replica.end('critic-6'), 'critic-6')
        with self.assertRaises(ValueError):
            replica.begin('critic-7')


if __name__ == '__main__':
    unittest.main()
