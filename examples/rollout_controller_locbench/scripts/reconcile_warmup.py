"""Derive critic-only targets under explicit terminal compaction failure semantics.

Never retry a model failure and never invent its token/logprob record. Existing
successful traces are unchanged. A typed incomplete-compaction event supervises
only exact earlier checkpoints with its terminal zero outcome. The failed reply's
text journal is retained as evidence, but it cannot be used as an actor sample.
"""
import argparse
import gzip
import json
from pathlib import Path
import pickle
import shutil
import time

from runtime_v2 import EXPERIMENT, MODEL, CONTEXT_LIMIT, contract, context, digest, compaction_terminal, verify_harness
from collect_warmup import write, save_native
from examples.locbench.dataset import load_cases
from examples.locbench.env import LocBenchEnv
from examples.locbench.metrics import score
from examples.locbench.repository import prepare

ERROR="IncompleteCompactionError('Incomplete compaction reply; refusing to replace task context')"


def reconcile(source,output):
    from transformers import AutoTokenizer
    verify_harness()
    if output.exists():raise ValueError('Derived corpus must use a fresh output directory')
    old=json.loads((source/'manifest.json').read_text())
    for name,sha in old['sources'].items():
        if digest(name)!=sha:raise ValueError('Original collection code changed')
    import runtime as original
    if old['environment']!=original.contract():raise ValueError('Source environment changed')
    if old['split_sha256']!=digest(EXPERIMENT/'data/split.json'):raise ValueError('Source split changed')
    output.mkdir(parents=True)
    manifest=dict(old,protocol='locbench-mc-critic-v2',environment=contract(),
        sources={str(p):digest(p) for p in (Path(__file__).resolve(),EXPERIMENT/'scripts/runtime_v2.py')},
        source_collection=str(source.resolve()),source_manifest_sha256=digest(source/'manifest.json'),
        derivation='Unchanged successful native traces; typed model compaction failures supervise only exact prior checkpoints with zero terminal recall')
    write(output/'manifest.json',manifest)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    cases={c.id:c for lane in ('train','development') for c in load_cases(EXPERIMENT/'data'/f'{lane}.jsonl')}
    counts=dict(completed=0,model_compaction_failures=0)
    for lane,key in [('train','train_ids'),('validation','validation_ids')]:
        for q in old[key]:
            for sample in range(old['samples_per_question']):
                src=source/lane/q/f'sample-{sample}.json';dst=output/lane/q/src.name
                dst.parent.mkdir(parents=True,exist_ok=True)
                failure=src.with_suffix('.failure.json')
                evidence={}
                if src.exists():
                    if failure.exists():raise ValueError('Ambiguous success/failure identity')
                    summary=json.loads(src.read_text())
                    if summary['manifest_sha256']!=digest(source/'manifest.json'):raise ValueError('Foreign source summary')
                    for suffix,key_ in [('.pkl.gz','source_sha256'),('.contexts.jsonl.gz','contexts_sha256')]:
                        if digest(src.with_suffix(suffix))!=summary[key_]:raise ValueError('Source payload checksum changed')
                        shutil.copyfile(src.with_suffix(suffix),dst.with_suffix(suffix))
                        evidence[str(src.with_suffix(suffix))]=digest(src.with_suffix(suffix))
                    evidence[str(src)]=digest(src)
                    counts['completed']+=1
                else:
                    if not failure.exists() or json.loads(failure.read_text())['error']!=ERROR:
                        raise ValueError(f'Missing or non-model outcome: {src}')
                    partial=src.with_suffix('.partial.pkl.gz');journal=src.with_suffix('.generations.jsonl')
                    saved=pickle.loads(gzip.decompress(partial.read_bytes()))
                    events=[json.loads(line) for line in journal.read_text().splitlines()]
                    if (not events or 'text' not in events[-1] or not events[-1].get('exact')
                        or events[-1]['completion_tokens']<=0 or events[-1]['sampling']['temperature']!=.7):
                        raise ValueError('Typed compaction failure lacks a completed generation journal')
                    checkpoints=saved['checkpoints'];prefix=saved['state'];case=cases[q]
                    if prefix.done or not checkpoints or any(p.done for p in checkpoints):raise ValueError('Invalid failure checkpoints')
                    if any(not t.exact for t in prefix.turns):raise ValueError('Failure prefix is inexact')
                    # No generated tokens are synthesized. This native container is
                    # explicitly critic-only and points to the raw rejected reply.
                    final=compaction_terminal(prefix,(),('recorded_incomplete_compaction',),'locbench-base-9b-mc-v1')
                    save_native(dst.with_suffix('.pkl.gz'),dict(checkpoints=checkpoints,final=final,
                        critic_only=True,rejected_reply_native_tokens_available=False))
                    repo=prepare(Path('/weka/scratch/jhu/bvandur1/zjiang31/rollout-controller/data/locbench/repositories'),case.repo,case.base_commit)
                    tools=LocBenchEnv(repo,case.gold,call_budget=20,maximum=10).schemas
                    targets=[]
                    for i,prefix_ in enumerate(checkpoints):
                        text=context(prefix_,tools,tokenizer);tokens=len(tokenizer.encode(text,add_special_tokens=False))+1
                        if tokens>CONTEXT_LIMIT:raise ValueError('Failure checkpoint exceeds context')
                        targets.append(dict(context=text,target=0.,group_index=q,turn=prefix_.turns_taken,
                            folds=prefix_.folds,tokens=tokens,metadata=dict(lane='critic',node_id=f'{sample}/{i}',
                                target_source='monte_carlo_suffix',diagnostics=dict(observations=1,mean_return=0.))))
                    dst.with_suffix('.contexts.jsonl.gz').write_bytes(gzip.compress(''.join(json.dumps(r)+'\n' for r in targets).encode(),mtime=0))
                    for evidence_file in (failure,partial,journal):evidence[str(evidence_file)]=digest(evidence_file)
                    metrics=score((),case.gold)
                    if metrics['reward']!=0:raise ValueError('Empty submission is not a zero outcome')
                    summary=dict(case_id=q,lane=lane,sample=sample,done=True,metrics=metrics,locations=[],
                        turns=final.turns_taken,folds=final.folds,checkpoints=len(targets),
                        source_sha256=digest(dst.with_suffix('.pkl.gz')),contexts_sha256=digest(dst.with_suffix('.contexts.jsonl.gz')),
                        seconds=saved['seconds']+events[-1]['elapsed_seconds'],critic_only=True,
                        termination='model_incomplete_compaction',rejected_reply_native_tokens_available=False)
                    counts['model_compaction_failures']+=1
                summary['manifest_sha256']=digest(output/'manifest.json')
                summary['source_evidence_sha256']=evidence
                write(dst,summary)
    write(output/'collection-complete.json',dict(traces=sum(counts.values()),time=time.time(),outcomes=counts))
    return counts


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();print(json.dumps(reconcile(args.source,args.output)),flush=True)
