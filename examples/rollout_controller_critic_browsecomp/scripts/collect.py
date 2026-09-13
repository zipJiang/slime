"""Fresh, bounded base-policy episodes and Monte Carlo checkpoint supervision."""
import argparse
import asyncio
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / 'snapshots/harness'))
from examples.browsercomp_plus.env import RetrievalArchive, load_cases, make_runner, grade
from examples.browsercomp_plus.pipeline import client_class
from examples.browsercomp_plus.judge import SYSTEM, TEMPLATE
from examples.deontic.synthetic_compaction import FoldGenerator, FOLD_DRAW
from step_controller.codec import ChatCodec
from step_controller.generation import PolicyFormat, SamplingParams
from step_controller.generation.parsing import QwenXMLToolCallParser
from context_bound import ContextBoundedCompactor, FOLD_OVERFLOW_LIMIT
from judge_contract import verdict
from provenance import sha256, validate_source_transition


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def context(payload, tools, tokenizer):
    state = dict(messages=list(payload.messages), tools=tools,
        workspace=dict(payload.workspace.snapshot()),
        remaining_task_steps=max(0, 48-payload.turns_taken),
        calls_remaining=payload.state.calls_remaining, calls_max=payload.state.calls_max,
        done=payload.done, truncated=payload.truncated)
    content = 'Predict the probability that a continuation from this checkpoint succeeds.\n'
    content += json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return tokenizer.apply_chat_template([dict(role='user', content=content)],
        tokenize=False, add_generation_prompt=True)


async def collect_all(episodes):
    # An episode failure must not cancel unrelated, expensive in-flight traces.
    results = await asyncio.gather(*episodes, return_exceptions=True)
    errors = [result for result in results if isinstance(result, Exception)]
    for result in results:
        if isinstance(result, BaseException) and not isinstance(result, Exception):
            raise result
    if errors:
        raise ExceptionGroup('Collection episodes failed after draining remaining work', errors)


async def main(args):
    import httpx
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    codec = ChatCodec(tokenizer)
    split = json.loads((EXPERIMENT/'data/split.json').read_text())
    assert hashlib.sha256(args.cases.read_bytes()).hexdigest() == split['cases_sha256']
    cases = load_cases(args.cases)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(actor=args.model, actor_revision=Path(args.model).name,
        split_sha256=hashlib.sha256((EXPERIMENT/'data/split.json').read_bytes()).hexdigest(),
        train_ids=split['train'][:128], validation_ids=split['development'][:32],
        samples_per_question=4, task_limit=48, max_context=32768, prompt_limit=14336,
        fold_overflow_context=FOLD_OVERFLOW_LIMIT,
        actor_max_tokens=6144, fold_max_tokens=4096, actor_temperature=1., actor_top_p=1.,
        fold_temperature=.2, fold_top_p=.95, call_budget=10, top_k=5,
        snippet_chars=700, read_chars=6000, checkpoint_sampling='root and every fold',
        target='empirical terminal outcome from fresh base-policy continuation',
        judge_model='Qwen/Qwen3.5-27B', judge_thinking=False, judge_output_regex='(EQUIVALENT|DIFFERENT)',
        judge_prompt_sha256=hashlib.sha256((SYSTEM+TEMPLATE).encode()).hexdigest(),
        sources={str(p.relative_to(EXPERIMENT)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in [*sorted((EXPERIMENT/'snapshots/harness').rglob('*.py')),
                     *(EXPERIMENT/'scripts'/name for name in ['collect.py','context_bound.py','judge_contract.py','provenance.py'])]})
    if (out/'manifest.json').exists():
        old=json.loads((out/'manifest.json').read_text())
        # Other scripts can be added while collection runs; the collector's own
        # immutable snapshot and protocol must stay identical on resume.
        for key in manifest.keys()-{'sources'}: assert old[key] == manifest[key], key
        for key, value in old['sources'].items():
            assert hashlib.sha256((EXPERIMENT/key).read_bytes()).hexdigest() == value, key
        validate_source_transition(out)
    else:
        write(out/'manifest.json', manifest)
    generator = FoldGenerator(model=args.model, served_model='Qwen/Qwen3.5-9B',
        format=PolicyFormat('qwen_xml', codec, QwenXMLToolCallParser()),
        base_url=args.actor_url, api_key='EMPTY', timeout=1800,
        default_params=SamplingParams(max_tokens=6144, temperature=1., top_p=1.,
            top_k=-1, repetition_penalty=1, logprobs=1))
    sem = asyncio.Semaphore(args.concurrency)
    judge_sem = asyncio.Semaphore(8)
    Client = client_class(args.retriever_code)
    async with httpx.AsyncClient(timeout=180) as http, Client(args.retriever_url,
            max_inflight=16, timeout=30, max_retries=3) as client:
        async def judge(question, answers, submitted):
            if not submitted.strip(): return dict(correct=False, empty=True)
            prompt=TEMPLATE.format(question=question, reference=json.dumps(answers), submitted=submitted)
            async with judge_sem:
                for attempt in range(3):
                    try:
                        response=await http.post(args.judge_url+'/chat/completions', json=dict(
                            model='Qwen/Qwen3.5-27B', messages=[dict(role='system',content=SYSTEM),
                                dict(role='user',content=prompt)], temperature=0, max_tokens=512,
                            chat_template_kwargs=dict(enable_thinking=False),
                            structured_outputs=dict(regex='(EQUIVALENT|DIFFERENT)')))
                        response.raise_for_status()
                        choice=response.json()['choices'][0]
                        text=choice['message']['content']
                        correct=verdict(text, choice['finish_reason'])
                        return dict(correct=correct, response=text, finish_reason=choice['finish_reason'])
                    except Exception:
                        if attempt==2: raise
                        await asyncio.sleep(2)

        fixtures=[('What year?', ['1988-96'], '1988-1996', True),
                  ('What city?', ['Paris'], 'London', False),
                  ('What number?', ['42'], 'forty-two', True),
                  ('What city?', ['Paris'], 'Paris or London', False)]
        calibration=[]
        for q,a,s,expected in fixtures:
            result=await judge(q,a,s)
            calibration.append(dict(expected=expected, **result))
            assert result['correct']==expected, 'Judge calibration failed'
        write(out/'judge-calibration.json', calibration)

        async def one(lane, case_id, sample):
            async with sem:
                case=cases[case_id]
                path=out/lane/case_id/f'sample-{sample}.json'
                if path.exists(): return
                archive=RetrievalArchive(client)
                runner,replay,workspace,prompt,compactor,tools=make_runner(case,codec,generator,
                    archive=archive,max_steps=48)
                runner._compactor=ContextBoundedCompactor(compactor)
                replay.allow_live=True
                token=FOLD_DRAW.set((f'browsecomp-critic-v1/{lane}/{case_id}/{sample}',0))
                started=time.time()
                checkpoints=[]
                snapshots=[]
                try:
                    state=await runner.start(prompt,workspace=workspace)
                    def capture():
                        snapshot=state.snapshot()
                        text=context(snapshot,tools,tokenizer)
                        if len(tokenizer.encode(text,add_special_tokens=False))+1>32768:
                            raise ValueError('Critic checkpoint exceeds native context budget')
                        checkpoints.append(dict(context=text,turn=state.turns_taken,folds=state.folds))
                        snapshots.append(snapshot)
                    capture()
                    while not state.done and not state.truncated and state.turns_taken<48:
                        before=len(state.turns)
                        state=await runner.advance(state)
                        if len(state.turns)==before: break
                        if not state.done and not state.truncated and state.turns[-1].tag=='fold': capture()
                    if not state.done and not state.truncated: state=await runner.finish(state)
                    if archive.failure is not None: raise RuntimeError('Retrieval infrastructure failed')
                    # Runner.finish marks the task horizon as truncated even
                    # when Env.finish produces a legitimate terminal outcome.
                    if not state.done: raise RuntimeError('No terminal outcome; no critic target created')
                    submitted=state.state.answer or ''
                    result=await judge(case.question,case.answers,submitted)
                    path.parent.mkdir(parents=True,exist_ok=True)
                    raw=gzip.compress(pickle.dumps(dict(checkpoints=snapshots,final=state.snapshot())),mtime=0)
                    path.with_suffix('.pkl.gz').write_bytes(raw)
                    archive.save(path.with_suffix('.retrieval.json.gz'))
                    rows=[]
                    for i, checkpoint in enumerate(checkpoints):
                        target=float(result['correct'])
                        rows.append(dict(**checkpoint,target=target,group_index=case_id,
                            metadata=dict(lane='critic',node_id=f'{sample}/{i}',target_source='monte_carlo_suffix',
                                diagnostics=dict(observations=1,mean_return=target))))
                    with gzip.open(path.with_suffix('.contexts.jsonl.gz'),'wt') as stream:
                        for row in rows: stream.write(json.dumps(row)+'\n')
                    write(path,dict(case_id=case_id,sample=sample,lane=lane,judge=result,
                        collection_manifest_sha256=sha256(out/'manifest.json'),
                        exact_match=grade(submitted,case),done=state.done,horizon_finished=state.truncated,turns=state.turns_taken,
                        folds=state.folds,checkpoints=len(rows),seconds=time.time()-started,
                        source_sha256=hashlib.sha256(raw).hexdigest(),
                        contexts_sha256=hashlib.sha256(path.with_suffix('.contexts.jsonl.gz').read_bytes()).hexdigest(),
                        retrieval_sha256=hashlib.sha256(path.with_suffix('.retrieval.json.gz').read_bytes()).hexdigest()))
                    print(json.dumps(dict(event='trace_complete',lane=lane,case_id=case_id,sample=sample,
                        correct=result['correct'],folds=state.folds,seconds=time.time()-started)),flush=True)
                except Exception as exc:
                    write(path.with_suffix('.failure.json'),dict(error=repr(exc),seconds=time.time()-started))
                    raise
                finally: FOLD_DRAW.reset(token)

        jobs=[(lane,k,i) for lane,ids in [('train',manifest['train_ids']),('validation',manifest['validation_ids'])]
              for k in ids for i in range(4)]
        # Interleave held-out collection with training so validation is available
        # even if a resource deadline interrupts the full collection.
        jobs.sort(key=lambda j:hashlib.sha256('/'.join(map(str,j)).encode()).hexdigest())
        if args.pilot: jobs=jobs[:4]
        await collect_all(one(*job) for job in jobs)
        write(out/('pilot-complete.json' if args.pilot else 'collection-complete.json'),
              dict(traces=len(jobs),unix_time=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['model','actor-url','judge-url','retriever-url']: p.add_argument('--'+name,required=True)
    for name in ['cases','output','retriever-code']: p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--concurrency',type=int,default=24)
    p.add_argument('--pilot',action='store_true')
    asyncio.run(main(p.parse_args()))
