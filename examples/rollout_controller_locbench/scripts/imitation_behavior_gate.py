"""Paired, prespecified behavior screen before promoting the imitation actor."""
import argparse
import json
import math
from pathlib import Path
from runtime_v2 import digest
from collect_warmup import write

CRITERIA = dict(min_post_fold_repeat_reduction=.20, max_recall_drop=0.,
                max_turn_ratio=1., max_extra_horizon_hits=0, max_failures=0)


def summarize(rows):
    good = [r for r in rows if 'error' not in r]
    for row in good:
        if not math.isfinite(row['reward']) or not 0 <= row['reward'] <= 1:
            raise ValueError('Invalid recall')
    return dict(trajectories=len(rows), failures=len(rows)-len(good),
                mean_recall=sum(r['reward'] for r in good)/len(rows),
                total_turns=sum(r['total_turns'] for r in good),
                repeated_calls=sum(r['repeated_calls'] for r in good),
                post_fold_repeated_calls=sum(r['post_fold_repeated_calls'] for r in good),
                horizon_hits=sum(r['horizon_hit'] for r in good))


def compare(base, trained):
    def keyed(rows):
        result={(r['query_id'], r['sample']):r for r in rows}
        if not rows or len(result)!=len(rows):raise ValueError('Missing or duplicate behavior outcomes')
        return result
    if keyed(base).keys()!=keyed(trained).keys():raise ValueError('Unpaired behavior outcomes')
    before, after = summarize(base), summarize(trained)
    if before['failures']:raise ValueError('Baseline has unresolved failures')
    checks=dict(
        no_failures=after['failures']==0,
        recall_preserved=after['mean_recall']+1e-9>=before['mean_recall'],
        fewer_post_fold_repeats=after['post_fold_repeated_calls']<=.8*before['post_fold_repeated_calls'],
        no_more_total_repeats=after['repeated_calls']<=before['repeated_calls'],
        no_more_turns=after['total_turns']<=before['total_turns'],
        no_more_horizon_hits=after['horizon_hits']<=before['horizon_hits'])
    return dict(passed=all(checks.values()), checks=checks, base=before, trained=after,
                criteria=CRITERIA, scope='Selected development cohort screen; not an unbiased test-set estimate')


def audit(base, trained):
    read=lambda p:json.loads(p.read_text())
    bm,tm=read(base/'manifest.json'),read(trained/'manifest.json')
    for key in ('protocol','questions','samples','environment','seed_namespace','split_sha256','sources'):
        if bm[key]!=tm[key]:raise ValueError('Behavior setup changed: '+key)
    if bm['model']==tm['model']:raise ValueError('Comparison requires the trained actor')
    expected={(q,s) for q in bm['questions'] for s in range(bm['samples'])}
    rows=[]
    for root in (base,trained):
        data=read(root/'results.json');complete=read(root/'complete.json')
        if {(r['query_id'],r['sample']) for r in data}!=expected or len(data)!=len(expected):
            raise ValueError('Incomplete behavior cohort')
        if complete['trajectories']!=len(data) or complete['failures']!=sum('error' in r for r in data):
            raise ValueError('Behavior completion mismatch')
        rows.append(data)
    result=compare(*rows)
    result['evidence']={str(r/f):digest(r/f) for r in (base,trained)
                        for f in ('manifest.json','results.json','complete.json')}
    write(trained/'review.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--trained',type=Path,required=True);a=p.parse_args()
    print(json.dumps(audit(a.base,a.trained),indent=2))
