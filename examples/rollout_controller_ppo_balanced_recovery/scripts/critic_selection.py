"""Fit learned checkpoint values; terminal values are already fixed by the estimator."""
from collections import defaultdict


def learned_critic_rows(records):
    selected = []
    counts = defaultdict(lambda: dict(exported=0, learned=0, fixed_terminal=0))
    for record in records:
        meta = record['metadata']
        if meta['lane'] != 'critic' or type(meta.get('terminal_boundary')) is not bool:
            raise ValueError('Critic supervision requires an explicit native terminal boundary flag')
        group = record['group_index']
        counts[group]['exported'] += 1
        if meta['terminal_boundary']:
            if record['target'] != 0. or meta['diagnostics']['mean_return'] != 0.:
                raise ValueError('A fixed terminal boundary must have zero future return')
            counts[group]['fixed_terminal'] += 1
        else:
            selected.append(record)
            counts[group]['learned'] += 1
    if not counts or any(c['learned'] == 0 for c in counts.values()):
        raise ValueError('Every question must provide a learned nonterminal critic target')
    return selected, dict(rule='nonterminal_only',
        reason='RefinedTdEstimator fixes terminal checkpoint values at zero without using the network.',
        exported=len(records), learned=len(selected), fixed_terminal=len(records)-len(selected),
        groups={str(group): count for group,count in sorted(counts.items())})
