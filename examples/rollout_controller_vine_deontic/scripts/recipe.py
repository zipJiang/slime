RECIPE_ID = 'vine-ppo-balanced-k1-base-both-null-g4-v5-20260920'

# v5 resumes the last accepted v2 actor while reducing retained root
# trajectories from five to four.  Keep the original VinePPO contract of one
# disposable outcome rollout per live state so the baseline remains comparable
# to the requested rollout budget.  The library supports k>1 separately, but
# changing k is a different experiment.
RESUME_COMPATIBLE_RECIPE_IDS = frozenset({
    RECIPE_ID,
    'vine-ppo-balanced-k3-base-both-null-g4-v4-20260920',
    'vine-ppo-balanced-k1-base-both-null-g5-v3-deterministic-zero-filter-20260920',
    'vine-ppo-balanced-k1-base-both-null-g5-v2-20260919',
})

def target_source(estimator):
    if estimator != 'vine_ppo': raise ValueError(estimator)
    return 'independent_outcome_mc'
