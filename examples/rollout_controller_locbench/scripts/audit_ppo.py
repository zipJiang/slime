"""Recompute prepared actor/critic targets from native LocBench search evidence."""
import argparse
import gzip
import json
from pathlib import Path
import pickle
import sys
from runtime_active import EXPERIMENT, contract, digest, verify_harness
from ppo_runtime import validate_environment
# ppo_runtime imports legacy-compatible helpers.  Re-establish the selected
# robust package tree before binding preparation/export classes or unpickling.
from runtime_v3 import activate as activate_robust
activate_robust(force=True)
sys.path.append(str(EXPERIMENT/'snapshots/native-support-v1'))
from step_controller import DirectBranchTdEstimator
from step_controller.preparation import prepare_samples
from step_controller.export import to_samples
from step_controller.reward.config import RewardConfig
from targets import split_targets
from examples.locbench.dataset import load_cases
from examples.locbench.metrics import score


def read_rows(path):
    with gzip.open(path,'rt') as stream:return [json.loads(line) for line in stream]


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def audit(directory):
    verify_harness();directory=Path(directory)
    recipe=json.loads((directory/'contract.json').read_text())
    summary=json.loads((directory/'summary.json').read_text())
    validate_environment(recipe)
    if recipe['context_source_sha256']!=digest(EXPERIMENT/'scripts/runtime_v2.py'):
        raise ValueError('PPO environment differs from the frozen critic context')
    if recipe['split_sha256']!=digest(EXPERIMENT/'data/split.json') or recipe['cases_sha256']!=digest(EXPERIMENT/'data/train.jsonl'):
        raise ValueError('PPO dataset changed')
    split=json.loads((EXPERIMENT/'data/split.json').read_text());selected=recipe['case_ids']
    if len(set(selected))!=len(selected) or not set(selected)<=set(split['train']):raise ValueError('Invalid training cohort')
    if len(summary['results'])!=len(selected):raise ValueError('Incomplete question batch')
    cases={c.id:c for c in load_cases(EXPERIMENT/'data/train.jsonl')}
    counts=dict(actor_edges=0,actor_spans=0,critic_checkpoints=0,empty_actor_groups=0)
    rc=RewardConfig(value_version=recipe['value_version'],value_prior_strength=recipe['prior_strength'],
        config_id=f"locbench-ppo-kappa-{recipe['prior_strength']:g}")
    for group,(question,result) in enumerate(zip(selected,summary['results'],strict=True)):
        if result['group_index']!=group or result['query_id']!=question:raise ValueError('Question order changed')
        stem=f'group-{group:03d}'
        native=directory/f'{stem}.native.pkl.gz';stored=directory/f'{stem}.prepared.pkl.gz'
        if digest(native)!=result['native_sha256'] or digest(stored)!=result['prepared_sha256']:
            raise ValueError('Saved training evidence changed')
        with gzip.open(native,'rb') as stream:state=pickle.load(stream)
        if state.stats.get('failures',0) or state.stats.get('score_failures',0):raise ValueError('Failed search batch')
        batch=prepare_samples(state,estimator=DirectBranchTdEstimator(),reward_config=rc,
            behavior_version=recipe['policy_version'])
        with gzip.open(stored,'rb') as stream:prepared=pickle.load(stream)
        if canonical(to_samples(batch,group_index=group))!=canonical(to_samples(prepared,group_index=group)):
            raise ValueError('Prepared source disagrees with native TD recomputation')
        actor,critic=split_targets(to_samples(batch,group_index=group,min_abs_advantage=recipe['actor_min_abs_advantage']))
        for records,lane in ((actor,'actor'),(critic,'critic')):
            for row in records:
                row['metadata'].update(query_id=question,policy_version=recipe['policy_version'],
                    server_weight_version=recipe['server_weight_version'])
                if lane=='critic':row['metadata'].update(
                    terminal_boundary=state.nodes[row['metadata']['node_id']].payload.done,target_source='direct_branch_mean')
            if canonical(records)!=canonical(read_rows(directory/f'{stem}.{lane}.jsonl.gz')):
                raise ValueError(f'{lane} targets disagree with native recomputation')
        terminals=[n.payload for n in state.nodes.values() if n.payload.done]
        rewards=[float(p.reward_outcome) for p in terminals]
        if rewards!=result['terminal_rewards'] or any(score(p.state.locations,cases[question].gold)['reward']!=r for p,r in zip(terminals,rewards,strict=True)):
            raise ValueError('Terminal outcome mismatch')
        counts['actor_edges']+=len({r['metadata']['node_id'] for r in actor})
        counts['actor_spans']+=len(actor);counts['critic_checkpoints']+=len(critic)
        counts['empty_actor_groups']+=not actor
    result=dict(passed=True,questions=len(selected),**counts,contract_sha256=digest(directory/'contract.json'))
    (directory/'target-replay-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args()
    print(json.dumps(audit(args.directory)),flush=True)
