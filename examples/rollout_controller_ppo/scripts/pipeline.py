"""Batch-boundary policy for one-update prefetch; no Ray or controller dependency."""
from dataclasses import dataclass


@dataclass(frozen=True)
class BatchStamp:
    collection_round: int
    behavior_round: int
    warmup_rounds: int

    def lineage(self, learner_round, *, overlap):
        actor = lambda r: max(0, r-self.warmup_rounds)
        actor_lag = actor(learner_round)-actor(self.behavior_round)
        critic_lag = learner_round-self.behavior_round
        maximum = 1 if overlap else 0
        if learner_round != self.collection_round or not 0 <= critic_lag <= maximum:
            raise ValueError('Critic snapshot lag or collection cursor violation')
        if not 0 <= actor_lag <= maximum:
            raise ValueError('Actor snapshot lag violation')
        return dict(collection_round=self.collection_round,
                    behavior_round=self.behavior_round, learner_round=learner_round,
                    actor_lag=actor_lag, critic_lag=critic_lag,
                    denominator='stored_behavior_logprobs', execution='overlap' if overlap else 'sync')


def save_boundary(round_id, warmup_rounds, save_interval, last_round):
    updates = max(0, round_id+1-warmup_rounds)
    return (round_id == 0 or round_id+1 == warmup_rounds or round_id == last_round or
            (updates > 0 and updates % save_interval == 0))


def may_prefetch(round_id, *, warmup_rounds, last_round, save_now, eval_now, stop_requested):
    # A checkpoint's question cursor includes exactly the trained batches. Never
    # save while an untrained prefetched batch has advanced the dataset cursor.
    return (round_id >= warmup_rounds and round_id < last_round and
            not save_now and not eval_now and not stop_requested)
