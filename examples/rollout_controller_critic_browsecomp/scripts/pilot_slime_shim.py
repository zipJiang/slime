"""Connect Slime's rollout manager to the BrowserComp zero-warmup collector."""
import gzip
import json
import os
from pathlib import Path
import subprocess

from batches import training_data

EXPERIMENT=Path(__file__).resolve().parents[1]
ROOT=EXPERIMENT.parents[2]


def read_rows(path):
    with gzip.open(path,'rt') as stream: return [json.loads(line) for line in stream]


def expected_batch(schedule,rollout_id,batch_size):
    if (schedule.get('schema')!='browsecomp-zero-warmup-pilot-schedule-v1'
            or schedule.get('updates')<2 or schedule.get('batch_size')!=batch_size):
        raise ValueError('Invalid zero-warmup pilot schedule')
    batches=schedule['batches']
    if not 0<=rollout_id<len(batches) or len(batches[rollout_id])!=batch_size:
        raise ValueError('Pilot rollout is outside the frozen two-update schedule')
    return batches[rollout_id]


def selected_questions(originals):
    questions=[]
    for group in originals:
        if len(group)!=1 or 'query_id' not in group[0].metadata:
            raise ValueError('Each pilot data group must contain one query identity')
        questions.append(str(group[0].metadata['query_id']))
    return questions


def convert_actor_data(args,samples):
    rows=[sample.metadata['prepared_record'] for sample in samples]
    return training_data(rows,lane='actor',expected_groups=range(args.rollout_batch_size))


def generate_rollout(args,rollout_id,data_source,evaluation=False):
    if evaluation:
        raise ValueError('The two-update warmstart pilot does not run final-test evaluation')
    from slime.rollout.base_types import RolloutFnTrainOutput
    from slime.utils.types import Sample
    run=Path(args.save).parent
    freeze=json.loads((run/'collection-freeze.json').read_text())
    if freeze['rollout_id']!=rollout_id:
        raise ValueError('Collector freeze does not match current pilot round')
    originals=data_source.get_samples(args.rollout_batch_size)
    questions=selected_questions(originals)
    schedule=json.loads(Path(args.pilot_schedule_audit).read_text())
    if questions!=expected_batch(schedule,rollout_id,args.rollout_batch_size):
        raise ValueError('Slime data cursor differs from the frozen pilot batch')
    directory=run/'pilot-rollouts'/f'train-{rollout_id:04d}'
    if directory.exists():
        raise ValueError('Refusing stale or partial pilot rollout reuse')
    question_path=run/f'questions-{rollout_id:04d}.json'
    question_path.write_text(json.dumps(questions)+'\n')
    command=[str(ROOT/'rollout-controller/.venv/bin/python'),str(EXPERIMENT/'scripts/pilot_collect.py'),
        '--checkpoint',args.hf_checkpoint,'--cases',args.pilot_cases,
        '--questions',str(question_path),'--output',str(directory),
        '--retriever-code',args.pilot_retriever_code,
        '--infrastructure-manifest',str(args.pilot_infrastructure_manifest),
        '--url',f'http://{args.sglang_router_ip}:{args.sglang_router_port}',
        '--value-url',freeze['value_url'],'--judge-url',args.pilot_judge_url,
        '--retriever-url',args.pilot_retriever_url,'--policy-version',freeze['policy_version'],
        '--server-weight-version',freeze['server_weight_version'],
        '--value-version',freeze['value_version'],'--seed-namespace',freeze['seed_namespace'],
        '--batch-size',str(args.rollout_batch_size),'--pass-tokens',str(args.pilot_pass_tokens),
        '--max-pass-attempts',str(args.pilot_max_pass_attempts),
        '--concurrency',str(args.pilot_search_concurrency),
        '--prior-strength',str(args.pilot_prior_strength)]
    with (run/f'collector-{rollout_id:04d}.log').open('x') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,
            cwd=EXPERIMENT/'snapshots/harness',
            env=dict(os.environ,PYTHONPATH=str(EXPERIMENT/'snapshots/harness')))
    summary=json.loads((directory/'summary.json').read_text())
    if (summary['policy_version']!=freeze['policy_version']
            or summary['server_weight_version']!=freeze['server_weight_version']
            or summary['value_version']!=freeze['value_version']):
        raise ValueError('Pilot collection used a different frozen model version')
    samples=[]
    for group,question in enumerate(questions):
        for row in read_rows(directory/f'group-{group:03d}.actor.jsonl.gz'):
            sample=Sample(index=len(samples),rollout_id=group,group_index=group,
                prompt=question,tokens=row['tokens'],response_length=row['response_length'],
                loss_mask=row['loss_mask'],rollout_log_probs=row['rollout_log_probs'],
                reward=row['reward'],metadata=dict(prepared_record=row),
                status=Sample.Status.COMPLETED)
            sample.weight_versions=[freeze['server_weight_version']]
            samples.append(sample)
    terminals=sum(r['terminal_count'] for r in summary['results'])
    if terminals<=0: raise ValueError('Pilot search produced no terminal outcomes')
    metrics={f'collection_{key}':value for key,value in summary['cost'].items()}
    metrics.update(critic_requests=summary['critic_requests'],critic_contexts=summary['critic_contexts'],
        search_terminals=terminals,
        search_terminal_success=sum(r['terminal_correct'] for r in summary['results'])/terminals)
    return RolloutFnTrainOutput(samples=samples,metrics=metrics)


__all__=['convert_actor_data','generate_rollout']
