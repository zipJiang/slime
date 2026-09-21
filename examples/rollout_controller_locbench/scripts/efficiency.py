"""Compare whole-edge cutoffs on identical prepared trees, without running models.

Sequence work is a proxy, not a wall-time claim. Native timings must be supplied
separately before recommending a cutoff or a GPU role split.
"""
import argparse
import gzip
import json
import math
import os
from pathlib import Path
import pickle
import statistics

if os.environ.get("LOC_COLLECTION_PROFILE") == "robust-null-v3":
    # This utility also runs as a fresh subprocess.  Normalize here rather
    # than relying on the parent's import order or PYTHONPATH precedence.
    from runtime_v3 import activate as activate_robust

    activate_robust(force=True)

from step_controller.export import to_samples

EXPERIMENT = Path(__file__).resolve().parents[1]

CUTOFFS = (None, 0., .01, .02, .04, .08, .12)


def choose_timing(timings, *, minimum_headroom_bytes=0):
    """Prefer less filtering within 5% of the fastest memory-eligible run."""
    eligible=[r for r in timings if r.get('min_allocator_headroom_bytes',0)>=minimum_headroom_bytes]
    if not eligible:
        raise ValueError('No PPO benchmark has the required allocator headroom')
    if any(not math.isfinite(r['overlapped_seconds']) or r['overlapped_seconds']<=0 for r in eligible):
        raise ValueError('Invalid native benchmark timing')
    fastest=min(r['overlapped_seconds'] for r in eligible)
    close=[r for r in eligible if r['overlapped_seconds']<=fastest*1.05]
    return min(close,key=lambda r:float('-inf') if r['cutoff'] is None else r['cutoff'])


def quantiles(values):
    ordered = sorted(values)
    if not ordered:
        return {}
    return {str(q): ordered[round(q * (len(ordered) - 1))]
            for q in (0., .1, .25, .5, .75, .9, .95, .99, 1.)}


def analyze(groups, cutoffs=CUTOFFS):
    """Use original question/edge weighting; tag mixed task/fold edges separately."""
    if not groups:
        raise ValueError('At least one complete question tree is required')
    edges = []
    for group, prepared in enumerate(groups):
        if prepared.failures:
            raise ValueError('Failed search cannot select efficiency settings')
        for edge in prepared.actor:
            if not math.isfinite(edge.weight) or edge.importance != 1.:
                raise ValueError('Efficiency pilot requires finite anchor-only advantages')
            tags = sorted({s.tag for s in edge.spans})
            edges.append(dict(group=group, node=edge.node_id, advantage=edge.weight,
                mass=abs(edge.weight) / len(prepared.actor) / len(groups),
                kind='+'.join(tags), spans=len(edge.spans),
                retain_for_training=any(span.retain_for_training for span in edge.spans),
                trainable_tokens=edge.edge_tokens,
                sequence_tokens=sum(len(s.tokens) for s in edge.spans),
                sequence_squared=sum(len(s.tokens)**2 for s in edge.spans)))
    total_mass = sum(e['mass'] for e in edges)
    candidates = []
    for cutoff in cutoffs:
        exports = [to_samples(p, group_index=g, min_abs_advantage=cutoff)
                   for g, p in enumerate(groups)]
        # The library deliberately retains malformed-compaction edges even when
        # their shaping advantage is below the numeric cutoff.  The analysis must
        # model that semantic exception as well as the threshold itself.
        exported = {(r['group_index'], r['metadata']['node_id'])
                    for rows in exports for r in rows if r['metadata']['lane'] == 'actor'}
        known = {(e['group'], e['node']) for e in edges}
        if not exported <= known:
            raise ValueError('Library export contains an unknown actor edge')
        expected = {(e['group'], e['node']) for e in edges
                    if cutoff is None or abs(e['advantage']) > cutoff
                    or e['retain_for_training']}
        # The exported records are what the native benchmark and trainer will
        # consume, so use them as the source of truth for work and mass.  Keep
        # any semantic difference visible instead of discarding an otherwise
        # valid, immutable rollout batch at this diagnostic boundary.
        kept = [e for e in edges if (e['group'], e['node']) in exported]
        semantic_mismatches = sorted(exported ^ expected)
        for g, rows in enumerate(exports):
            reference = to_samples(groups[g], group_index=g)
            if ([r for r in rows if r['metadata']['lane'] == 'critic'] !=
                    [r for r in reference if r['metadata']['lane'] == 'critic']):
                raise ValueError('Filtering changed critic supervision')
            if any(r['metadata']['group_edge_count'] != len(groups[g].actor)
                   for r in rows if r['metadata']['lane'] == 'actor'):
                raise ValueError('Filtering changed the original edge denominator')
        mass = sum(e['mass'] for e in kept)
        candidates.append(dict(cutoff=cutoff, edges=len(kept), removed_edges=len(edges)-len(kept),
            export_semantic_mismatches=[dict(group=g,node=n) for g,n in semantic_mismatches],
            actor_empty_groups=sorted(set(range(len(groups)))-{e['group'] for e in kept}),
            weighted_absolute_advantage_mass=mass,
            retained_mass_fraction=mass/total_mass if total_mass else None,
            positive_edges=sum(e['advantage'] > 0 for e in kept),
            negative_edges=sum(e['advantage'] < 0 for e in kept),
            **{key: sum(e[key] for e in kept) for key in
               ('spans', 'trainable_tokens', 'sequence_tokens', 'sequence_squared')},
            by_kind={kind: dict(edges=sum(e['kind']==kind for e in kept),
                trainable_tokens=sum(e['trainable_tokens'] for e in kept if e['kind']==kind),
                weighted_absolute_advantage_mass=sum(e['mass'] for e in kept if e['kind']==kind))
                for kind in sorted({e['kind'] for e in edges})}))
    return dict(questions=len(groups), original_edges=len(edges),
        critic_checkpoints=sum(len(p.critic) for p in groups),
        advantage_quantiles=quantiles([e['advantage'] for e in edges]),
        absolute_advantage_quantiles=quantiles([abs(e['advantage']) for e in edges]),
        exactly_zero_advantage_edges=sum(e['advantage']==0 for e in edges),
        question_weighted_absolute_advantage_mass=total_mass,
        candidates=candidates,
        interpretation='Sequence tokens and squared lengths are work proxies; use native packed timings for speed.',
        selected_cutoff=None, selection_status='Awaiting native timings and task/fold coverage review')


def overlap_estimate(*, collection_seconds, actor_seconds, critic_seconds, publication_seconds):
    values = (collection_seconds, actor_seconds, critic_seconds, publication_seconds)
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Timings must be finite and nonnegative')
    learner = actor_seconds + critic_seconds
    return dict(learner_seconds=learner,
        steady_state_seconds=max(collection_seconds, learner)+publication_seconds,
        learner_wait_seconds=max(0., collection_seconds-learner),
        collection_wait_seconds=max(0., learner-collection_seconds),
        assumption='One batch of lookahead, separate inference GPUs, publication outside overlap; excludes save boundaries')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    summary = json.loads((args.collection/'summary.json').read_text())
    groups = []
    for row in summary['results']:
        path = args.collection/f"group-{row['group_index']:03d}.prepared.pkl.gz"
        with gzip.open(path, 'rb') as stream:
            groups.append(pickle.load(stream))
    result = analyze(groups)
    result['collection'] = dict(seconds=summary['seconds'], cost=summary['cost'],
        terminal_rewards=[v for r in summary['results'] for v in r['terminal_rewards']],
        zero_immediate_reward_edges=sum(r['zero_immediate_reward_edges'] for r in summary['results']))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
