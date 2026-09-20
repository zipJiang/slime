RECIPE_ID = 'locbench-vine-ppo-k1-imitation-v1-20260917'


def target_source(estimator):
    if estimator != 'vine_ppo':
        raise ValueError(estimator)
    return 'independent_outcome_mc'
