"""Owned subprocess collection; native Slime owns engines, this driver owns questions."""
import gzip
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from runtime_active import EXPERIMENT,HARNESS,MODEL,digest

PY='/weka/scratch/jhu/bvandur1/zjiang31/rollout-controller/.venv/bin/python'


def unused_rollout(*args,**kwargs):
    raise RuntimeError('LocBench driver owns its frozen question schedule; manager.generate is not used')


def read_rows(path):
    with gzip.open(path,'rt') as stream:return [json.loads(line) for line in stream]


def router_address(manager):
    # Deployment mutates the manager's deserialized args, not the driver's copy.
    return _router_address(manager.args)


def _router_address(args):
    host,port=args.sglang_router_ip,args.sglang_router_port
    if not isinstance(host,str) or not host or type(port) is not int or not 1<=port<=65535:
        raise ValueError('Rollout manager has no active router endpoint')
    return dict(host=host,port=port)


def command(args,*,questions,output,frozen,cutoff,pass_tokens):
    endpoint=_router_address(args)
    result=[PY,str(EXPERIMENT/'scripts/collect_ppo.py'),'--checkpoint',str(getattr(args,'hf_checkpoint',MODEL)),
        '--questions',str(questions),'--output',str(output),
        '--url',f"http://{endpoint['host']}:{endpoint['port']}",
        '--value-url',frozen['value_url'],'--policy-version',frozen['policy_version'],
        '--server-weight-version',frozen['server_weight_version'],'--value-version',frozen['value_version'],
        '--seed-namespace',frozen['seed_namespace'],'--batch-size',str(args.rollout_batch_size),
        '--pass-tokens',str(pass_tokens),'--max-pass-attempts',str(args.loc_max_pass_attempts),
        '--concurrency',str(args.loc_search_concurrency),'--prior-strength',str(args.loc_prior_strength)]
    result+=['--collection-profile',getattr(args,'loc_collection_profile','original')]
    if cutoff is not None:result+=['--min-abs-advantage',str(cutoff)]
    return result


class Collection:
    def __init__(self,args,run,round_id,questions,frozen,*,cutoff,pass_tokens):
        self.directory=Path(run)/'pilot-rollouts'/f'train-{round_id:04d}'
        self.frozen=frozen;self.started=time.monotonic()
        selection=Path(run)/f'questions-{round_id:04d}.json'
        selection.write_text(json.dumps(questions)+'\n')
        self.log=(Path(run)/f'collector-{round_id:04d}.log').open('x')
        env=dict(os.environ,PYTHONPATH=str(HARNESS))
        self.process=subprocess.Popen(command(args,questions=selection,output=self.directory,
            frozen=frozen,cutoff=cutoff,pass_tokens=pass_tokens),stdout=self.log,stderr=subprocess.STDOUT,
            cwd=HARNESS,env=env,start_new_session=True)

    def finish(self):
        code=self.process.wait();self.log.close()
        if code:raise RuntimeError(f'LocBench collection failed: {self.directory}, exit={code}')
        value=json.loads((self.directory/'summary.json').read_text())
        if any(value[k]!=self.frozen[k] for k in ('policy_version','server_weight_version','value_version')):
            raise ValueError('Collector returned another behavior version')
        return dict(directory=self.directory,frozen=self.frozen,seconds=value['seconds'],summary=value)

    def cancel(self):
        if self.process.poll() is None:
            os.killpg(self.process.pid,signal.SIGTERM)
            try:self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(self.process.pid,signal.SIGKILL);self.process.wait()
        self.log.close()


def audit_collection(directory):
    log=Path(directory)/'target-replay-audit.log'
    with log.open('x') as stream:
        subprocess.run([PY,str(EXPERIMENT/'scripts/audit_ppo.py'),str(directory)],check=True,
            stdout=stream,stderr=subprocess.STDOUT,cwd=HARNESS,env=dict(os.environ,PYTHONPATH=str(HARNESS)))
    return json.loads((Path(directory)/'target-replay-audit.json').read_text())


def materialize(directory,cutoff):
    output=Path(directory)/'exports'/('none' if cutoff is None else str(cutoff))
    command=[PY,str(EXPERIMENT/'scripts/reexport_ppo.py'),'--source',str(directory),'--output',str(output)]
    if cutoff is not None:command+=['--min-abs-advantage',str(cutoff)]
    subprocess.run(command,check=True,cwd=HARNESS,env=dict(os.environ,PYTHONPATH=str(HARNESS)))
    report=json.loads((output/'export.json').read_text())
    if report['source_contract_sha256']!=digest(Path(directory)/'contract.json'):raise ValueError('Foreign cutoff export')
    records=[]
    for name,sha in sorted(report['files'].items()):
        if digest(output/name)!=sha:raise ValueError('Cutoff export checksum mismatch')
        records.extend(read_rows(output/name))
    return records,report


def replay_benchmark(args,run,plan,*,schedule_hash,behavior_version,value_url):
    """Reuse only an audited, untrained batch from the identical base behavior."""
    import shutil
    from ppo_runtime import environment
    expected_environment = environment(getattr(args, 'loc_collection_profile', 'original'))
    if args.loc_resume_run:
        raise ValueError('Saved base-policy collection cannot seed a resumed model')
    source=Path(args.loc_benchmark_source).resolve()
    producer_root=source.parent.parent
    producer=json.loads((producer_root/'recipe.json').read_text())
    recipe=json.loads((source/'contract.json').read_text())
    summary=json.loads((source/'summary.json').read_text())
    checkpoints=list((producer_root/'checkpoints').glob('round-*.json'))
    failure=json.loads((producer_root/'failed.json').read_text()) if (producer_root/'failed.json').exists() else None
    if (producer['candidate_sha256']!=digest(args.loc_candidate) or producer['schedule_sha256']!=schedule_hash
        or producer['environment']!=expected_environment or producer['resume'] is not None
        or recipe['environment']!=expected_environment or recipe['case_ids']!=plan['batches'][0]
        or recipe['pass_tokens']!=args.loc_pass_tokens or recipe['actor_min_abs_advantage'] is not None
        or recipe['policy_version']!='actor-0000' or recipe['value_version']!='critic-0000'
        or recipe['server_weight_version']!=behavior_version
        or recipe['seed_namespace']!=f'{args.loc_seed_namespace}/0000'
        or recipe['prior_strength']!=args.loc_prior_strength
        or recipe['max_pass_attempts']!=args.loc_max_pass_attempts
        or recipe['concurrency']!=args.loc_search_concurrency):
        raise ValueError('Saved benchmark batch has another base policy, critic, schedule or search recipe')
    if (checkpoints or (producer_root/'training-complete.json').exists()
        or (failure is not None and (failure.get('actor_updates') or failure.get('critic_updates')))):
        raise ValueError('Saved benchmark batch comes from a model that already took an optimizer step')
    if any(summary[k]!=recipe[k] for k in ('policy_version','value_version','server_weight_version')):
        raise ValueError('Saved benchmark summary has another behavior identity')
    target=Path(run)/'pilot-rollouts/train-0000'
    shutil.copytree(source,target,ignore=shutil.ignore_patterns('exports','failed.json','training-lineage.json',
        'training-complete.json','target-replay-audit.json','target-replay-audit.log'))
    frozen=dict(rollout_id=0,behavior_round=0,policy_version='actor-0000',server_weight_version=behavior_version,
        value_version='critic-0000',value_url=value_url,seed_namespace=f'{args.loc_seed_namespace}/0000')
    evidence=dict(source=str(source),source_recipe_sha256=digest(source.parent.parent/'recipe.json'),
        source_contract_sha256=digest(source/'contract.json'),source_summary_sha256=digest(source/'summary.json'),
        collection_rollout_gpus=producer['arguments']['rollout_num_gpus'],
        scope='Fresh base-policy benchmark/training seed; collection time comes from the original producer; targets are replay-audited again')
    (Path(run)/'benchmark-source.json').write_text(json.dumps(evidence,indent=2)+'\n')
    return dict(directory=target,frozen=frozen,seconds=summary['seconds'],summary=summary,replayed=True,
        collection_rollout_gpus=evidence['collection_rollout_gpus'])
