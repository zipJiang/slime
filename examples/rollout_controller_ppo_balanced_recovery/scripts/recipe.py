"""Versioned preparation contract for the isolated direct-branch ablation."""

RECIPE_ID = 'direct-branch-td-balanced-v2-base-warmup10-20260912'


def estimator_for_round(round_id, warmup_rounds):
    if round_id < 0 or warmup_rounds < 0:
        raise ValueError('Negative preparation cursor')
    return 'refined_td' if round_id < warmup_rounds else 'direct_branch_td'


def target_source(estimator):
    if estimator == 'refined_td':
        return 'warmup_empirical_suffix_mean'
    if estimator == 'direct_branch_td':
        return 'direct_branch_mean'
    raise ValueError(f'Unsupported preparation estimator: {estimator}')
