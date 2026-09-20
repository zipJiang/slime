"""Measure weak actor signals without filtering or rescaling any record."""


def summarize(records):
    placeholders = [r for r in records if r['metadata'].get('zero_advantage_placeholder')]
    records = [r for r in records if not r['metadata'].get('zero_advantage_placeholder')]
    edges = {}
    total_tokens = 0
    thresholds = (0., .001, .01, .05, .1)
    token_counts = dict.fromkeys(thresholds, 0)
    for row in records:
        identity = (row['group_index'], row['metadata']['node_id'])
        advantage = float(row['reward'])
        if identity in edges and edges[identity] != advantage:
            raise ValueError('Conflicting advantages on one edge')
        edges[identity] = advantage
        tokens = sum(row['loss_mask'])
        total_tokens += tokens
        for threshold in thresholds:
            if abs(advantage) <= threshold:
                token_counts[threshold] += tokens
    report = dict(edges=len(edges), tokens=total_tokens,
                  zero_advantage_placeholder_groups=len(placeholders),
                  positive_edges=sum(v > 0 for v in edges.values()),
                  negative_edges=sum(v < 0 for v in edges.values()))
    for threshold in thresholds:
        key = str(threshold).replace('.', '_')
        report[f'abs_le_{key}_edge_fraction'] = (
            sum(abs(v) <= threshold for v in edges.values()) / len(edges)
            if edges else 0.0
        )
        report[f'abs_le_{key}_token_fraction'] = (
            token_counts[threshold] / total_tokens if total_tokens else 0.0
        )
    return report
