"""Conservative, predeclared screening criteria; semantic repetition review remains required."""
from collections import defaultdict
from statistics import mean
from ppo_runtime import PROFILES


def evaluate(rows):
    groups={p:{(r['query_id'],r['sample']):r for r in rows if r['profile']==p} for p in PROFILES}
    keys=set(groups['original'])
    if len(rows)!=24 or len(keys)!=8 or any(set(g)!=keys for g in groups.values()):
        raise ValueError('Incomplete or duplicate paired pilot')
    totals={p:{key:sum(r[key] for r in group.values()) for key in
        ('seconds','reward','total_turns','tool_calls','repeated_calls','post_fold_repeated_calls','output_tokens','input_tokens','total_region_tokens','horizon_hit')}
        for p,group in groups.items()}
    for p,group in groups.items():totals[p]['max_region_tokens']=max(r['max_region_tokens'] for r in group.values())
    comparisons={}
    for p in PROFILES[1:]:
        baselines=['original'] if p=='reply8' else ['original','reply8']
        checks={}
        for baseline in baselines:
            a,b=totals[p],totals[baseline]
            checks[baseline]=dict(
                quality_noninferior=a['reward']>=b['reward']-1e-9,
                no_large_question_regression=all(
                    mean(r['reward'] for (qid,_),r in groups[p].items() if qid==q)>=
                    mean(r['reward'] for (qid,_),r in groups[baseline].items() if qid==q)-.25-1e-9
                    for q,_ in keys),
                wall_time_within_5_percent=a['seconds']<=b['seconds']*1.05,
                turns_within_5_percent=a['total_turns']<=b['total_turns']*1.05,
                tool_calls_within_5_percent=a['tool_calls']<=b['tool_calls']*1.05,
                no_more_exact_repeats=a['repeated_calls']<=b['repeated_calls'],
                no_more_post_fold_repeats=a['post_fold_repeated_calls']<=b['post_fold_repeated_calls'],
                no_more_horizon_hits=a['horizon_hit']<=b['horizon_hit'])
        exposed=sum(r['threshold_folds']>0 for r in groups[p].values())
        memory_reduction=totals[p]['max_region_tokens']<=totals['original']['max_region_tokens']*.9
        # reply8 can only demonstrate a behavior difference if the baseline actually exceeded 8K.
        treatment_exposed=(exposed>=4 if p=='compact16-reply8' else any(r['max_task_reply']>8192 for r in groups['original'].values()))
        passed=all(all(c.values()) for c in checks.values()) and memory_reduction and treatment_exposed
        comparisons[p]=dict(screen_passed=passed,checks=checks,memory_shape_reduced_10_percent=memory_reduction,
            treatment_exposed=treatment_exposed,trajectories_with_token_triggered_folds=exposed)
    selected=next((p for p in ('compact16-reply8','reply8') if comparisons[p]['screen_passed']),None)
    return dict(passed=selected is not None,selected_profile=selected,totals=totals,comparisons=comparisons,
        semantic_review_required=True,training_started=False,
        scope='Fresh development continuations; shape proxy is not a GPU-memory measurement; small sample screening, not a generalization claim')
