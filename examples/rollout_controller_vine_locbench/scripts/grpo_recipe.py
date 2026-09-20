RECIPE_ID = 'locbench-grpo-root8-imitation-zero-filter-v2-20260920'

RESUME_COMPATIBLE_RECIPE_IDS = frozenset({
    RECIPE_ID,
    'locbench-grpo-root8-imitation-v1-20260920',
})


def target_source(estimator):
    if estimator != 'grpo':
        raise ValueError(estimator)
    return 'root_group_outcome_standardized'
