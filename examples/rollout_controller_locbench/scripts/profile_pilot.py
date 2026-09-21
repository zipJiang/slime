"""Paired fresh development trajectories for compaction behavior screening, never training."""
import argparse
import asyncio
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import time

from runtime_v2 import EXPERIMENT, MODEL, TASK_LIMIT, digest, verify_harness
from ppo_runtime import PROFILES, environment, make_world
from collect_ppo import BoundedSlimePolicy, write
from examples.locbench.dataset import load_cases
from examples.locbench.repository import prepare
from examples.locbench.metrics import score
from step_controller.generation import SamplingParams, GenerateResult
from step_controller.generation.policy import PolicyFormat
from step_controller.harness.packing import regions

QUESTIONS = ['ray-project__ray-49118','zulip__zulip-31168','Qiskit__qiskit-13141','yt-dlp__yt-dlp-11750']


def repeated_calls(events):
    seen=set();before_fold=set();fold=0;repeats=[];calls=0
    for event in events:
        if event['tag']=='fold':
            before_fold=set(seen);fold+=1
            continue
        for call in event['calls']:
            if call['name']=='submit':continue
            calls+=1
            key=json.dumps(call,sort_keys=True,separators=(',',':'))
            if key in seen:
                repeats.append(dict(turn=event['turn'],fold=fold,after_fold=key in before_fold,call=call))
            seen.add(key)
    return dict(tool_calls=calls,repeated_calls=len(repeats),post_fold_repeated_calls=sum(r['after_fold'] for r in repeats),repeats=repeats)


def metrics(state, policy):
    events=[];calls_remaining=20;threshold_folds=0
    for i,turn in enumerate(state.turns):
        text=policy.decode(turn.tokens,skip_special_tokens=True)
        calls=[]
        if turn.tag=='task' and turn.tokens:
            parsed=policy.parse(GenerateResult(tokens=turn.tokens,text=text))
            for call in parsed.tool_calls:
                try:arguments=json.loads(call.arguments)
                except (ValueError,TypeError):arguments=call.arguments
                calls.append(dict(name=call.name,arguments=arguments))
        if turn.tag=='fold' and calls_remaining>0:threshold_folds+=1
        if turn.transition is not None:calls_remaining=turn.transition.next_state.calls_remaining
        elif turn.tag=='fold':calls_remaining=20
        events.append(dict(turn=i,tag=turn.tag,prompt_tokens=len(turn.prefix),reply_tokens=len(turn.tokens),calls=calls,text=text))
    packed=regions(state.turns)
    return dict(**repeated_calls(events),events=events,threshold_folds=threshold_folds,
        task_turns=sum(t.tag=='task' and bool(t.tokens) for t in state.turns),
        fold_generations=sum(t.tag=='fold' for t in state.turns),
        total_turns=sum(bool(t.tokens) for t in state.turns),folds=state.folds,
        output_tokens=sum(len(t.tokens) for t in state.turns),
        input_tokens=sum(len(t.prefix) for t in state.turns),
        max_region_tokens=max((len(p.tokens) for p in packed),default=0),
        total_region_tokens=sum(len(p.tokens) for p in packed),
        max_task_reply=max((len(t.tokens) for t in state.turns if t.tag=='task'),default=0),
        fold_prompt_tokens=[len(t.prefix) for t in state.turns if t.tag=='fold'])


async def run(args):
    import httpx
    from transformers import AutoTokenizer
    verify_harness();out=args.output;out.mkdir(exist_ok=False,parents=True)
    split=json.loads((EXPERIMENT/'data/split.json').read_text())
    if not set(QUESTIONS)<=set(split['development']):raise ValueError('Pilot must remain outside training/test')
    manifest=dict(protocol='locbench-compaction-behavior-pilot-v1',questions=QUESTIONS,samples=2,
        question_selection='Four development questions with long historical executions; workload stress cohort, not an unbiased quality estimate',
        policy_version=args.policy_version,server_weight_version=args.server_weight_version,
        environments={p:environment(p) for p in PROFILES},seed_namespace='locbench-compaction-behavior-v1',
        split_sha256=digest(EXPERIMENT/'data/split.json'),pilot_sha256=digest(__file__),
        caveat='Small screening study; exact repeated calls are flags for review, not automatic proof of unnecessary work')
    write(out/'manifest.json',manifest)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    fmt=PolicyFormat.resolve(MODEL,tokenizer=tokenizer,profile='qwen_xml')
    cases={c.id:c for c in load_cases(EXPERIMENT/'data/development.jsonl')}
    repos={q:await asyncio.to_thread(prepare,args.cache,cases[q].repo,cases[q].base_commit) for q in QUESTIONS}
    async with httpx.AsyncClient(timeout=1800.,limits=httpx.Limits(max_connections=32)) as http:
        async def one(profile,q,sample):
            cfg=environment(profile);draw=Counter();requests=[]
            stem=f'{profile}/{q}/sample-{sample}'
            async def post(url,payload):
                kind='fold' if payload['sampling_params']['temperature']==.7 else 'task'
                key=f"{manifest['seed_namespace']}/{q}/{sample}/{kind}/{draw[kind]}";draw[kind]+=1
                payload['sampling_params']['sampling_seed']=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%2**31
                t0=time.monotonic();response=await http.post(url,json=payload);response.raise_for_status();result=response.json()
                meta=result['meta_info']
                if str(meta.get('weight_version'))!=args.server_weight_version:raise ValueError('Pilot actor changed')
                requests.append(dict(kind=kind,input_tokens=len(payload['input_ids']),seconds=time.monotonic()-t0,
                    sampling_params=payload['sampling_params'],finish_reason=meta.get('finish_reason')))
                return result
            policy=BoundedSlimePolicy(post,args.url.rstrip('/')+'/generate',model=MODEL,format=fmt,
                version=args.policy_version,default_params=SamplingParams(max_tokens=cfg['actor_reply_limit'],temperature=.6,top_p=.95,top_k=20,logprobs=1))
            runner,workspace,prompt,_=make_world(cases[q],repos[q],policy,profile=profile)
            started=time.monotonic();state=await runner.start(prompt,workspace=workspace)
            for _ in range(TASK_LIMIT+2):
                if state.done or state.truncated:break
                before=len(state.turns);state=await runner.advance(state)
                write(out/(stem+'.progress.json'),dict(turns=state.turns_taken,folds=state.folds,seconds=time.monotonic()-started))
                if len(state.turns)==before:raise ValueError('Pilot made no progress')
            if not state.done:state=await runner.finish(state)
            if not state.done or not all(t.exact for t in state.turns):raise ValueError('Incomplete or inexact pilot')
            seconds=time.monotonic()-started
            reward=score(state.state.locations,cases[q].gold)['reward']
            if reward!=state.reward_outcome:raise ValueError('Terminal score mismatch')
            data=metrics(state,policy)
            if data['max_task_reply']>cfg['actor_reply_limit']:raise ValueError('Task cap not enforced')
            result=dict(profile=profile,query_id=q,sample=sample,seconds=seconds,reward=reward,
                submitted=state.state.submitted,horizon_hit=state.turns_taken+state.folds>=TASK_LIMIT,
                requests=requests,**data)
            native=out/(stem+'.native.pkl.gz');native.parent.mkdir(parents=True,exist_ok=True)
            native.write_bytes(gzip.compress(pickle.dumps(state.snapshot()),compresslevel=1,mtime=0))
            result['native_sha256']=digest(native);write(out/(stem+'.json'),result)
            print(json.dumps({k:result[k] for k in ['profile','query_id','sample','seconds','reward','total_turns','folds','repeated_calls','post_fold_repeated_calls','max_region_tokens']}),flush=True)
            return result
        results=[]
        # Reverse arm order for the second seed, limiting cache/order bias.
        for sample,profiles in enumerate((PROFILES,tuple(reversed(PROFILES)))):
            for profile in profiles:
                write(out/'status.json',dict(stage='collecting',sample=sample,profile=profile,completed=len(results)))
                rows=await asyncio.gather(*(one(profile,q,sample) for q in QUESTIONS),return_exceptions=True)
                errors=[repr(r) for r in rows if isinstance(r,BaseException)]
                if errors:
                    write(out/'failed.json',dict(errors=errors));raise RuntimeError(str(errors))
                results.extend(rows)
        # Preserve compact metrics; decoded events/requests remain in individual traces.
        summary=[{k:v for k,v in r.items() if k not in ('events','requests')} for r in results]
        write(out/'results.json',summary)
        from profile_gate import evaluate
        write(out/'comparison.json',evaluate(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    for name in ('url','policy-version','server-weight-version'):p.add_argument('--'+name,required=True)
    p.add_argument('--cache',type=Path,default=Path('/weka/scratch/jhu/bvandur1/zjiang31/rollout-controller/data/locbench/repositories'))
    asyncio.run(run(p.parse_args()))
