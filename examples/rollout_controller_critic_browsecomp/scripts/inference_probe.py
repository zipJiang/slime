"""Cover every held-out question and its longest saved compacted context."""
from collections import defaultdict


def probe_indices(rows, token_lengths):
    groups=defaultdict(list)
    for i,(row,length) in enumerate(zip(rows,token_lengths,strict=True)):
        if length <= 0: raise ValueError('Probe contexts must be nonempty')
        groups[row['group_index']].append(i)
    if not groups: raise ValueError('No validation questions to probe')
    selected=[]
    for indices in groups.values():
        roots=[i for i in indices if rows[i]['turn']==0]
        folds=[i for i in indices if rows[i]['turn']!=0]
        if len(roots)!=1: raise ValueError('Expected one aggregated root per validation question')
        selected.extend(roots)
        if folds: selected.append(max(folds,key=lambda i:token_lengths[i]))
    return selected
