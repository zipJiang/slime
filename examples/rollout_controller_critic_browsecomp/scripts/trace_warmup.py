"""Root-aware warmup weighting and validation-based checkpoint selection."""
from collections import defaultdict
import math

from batches import training_data
from dataset import metrics


def checkpoint_weights(rows, root_mass=.25):
    """Equal questions; roots receive 25% and fold states 75% within a question."""
    if not math.isfinite(root_mass) or not 0 < root_mass < 1:
        raise ValueError('Root mass must be strictly between zero and one')
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['group_index']].append(i)
    if not groups:
        raise ValueError('No question records')
    weights = [0.] * len(rows)
    for indices in groups.values():
        roots = [i for i in indices if rows[i]['turn'] == 0]
        folds = [i for i in indices if rows[i]['turn'] != 0]
        for stratum, mass in ((roots, root_mass), (folds, 1 - root_mass)):
            if not stratum:
                continue
            if not roots or not folds:
                mass = 1.
            for i in stratum:
                weights[i] = mass / len(stratum)
    return weights


def packet(records, root_mass=.25):
    data = training_data(records, lane='critic', expected_groups={r['group_index'] for r in records})
    weights = checkpoint_weights(records, root_mass)
    data['rollout_mask_sums'] = [1 / weight for weight in weights]
    return data


def baseline(rows, root_mass=.25):
    weights = checkpoint_weights(rows, root_mass)
    return sum(w * r['target'] for w, r in zip(weights, rows, strict=True)) / sum(weights)


def report(rows, predictions, constant, root_mass=.25):
    result = metrics(rows, predictions, constant)
    weights = checkpoint_weights(rows, root_mass)
    result['selection_mse'] = sum(w * (p - r['target']) ** 2 for w, r, p in
        zip(weights, rows, predictions, strict=True)) / sum(weights)
    result['selection_baseline_mse'] = sum(w * (constant - r['target']) ** 2 for w, r in
        zip(weights, rows, strict=True)) / sum(weights)
    result['root_mass'] = root_mass
    result['by_checkpoint'] = {}
    for name, root in [('root', True), ('fold', False)]:
        chosen = [(r, p) for r, p in zip(rows, predictions, strict=True) if (r['turn'] == 0) == root]
        if chosen:
            result['by_checkpoint'][name] = metrics([r for r, _ in chosen], [p for _, p in chosen], constant)
    return result


def choose_best(reports, *, min_delta=1e-4):
    """Keep the earliest meaningful improvement; include the starting critic."""
    if not reports or min_delta < 0 or not math.isfinite(min_delta):
        raise ValueError('Invalid checkpoint selection inputs')
    best = 0
    for i, item in enumerate(reports):
        score = item['selection_mse']
        if not math.isfinite(score):
            raise ValueError('Nonfinite validation score')
        if score < reports[best]['selection_mse'] - min_delta:
            best = i
    return best
