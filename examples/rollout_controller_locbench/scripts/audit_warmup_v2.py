"""Independently reconstruct every saved warmup target and checkpoint string."""
import argparse
import gzip
import json
from pathlib import Path
import pickle

from runtime_v2 import EXPERIMENT, MODEL, CONTEXT_LIMIT, contract, context, digest, verify_harness
from collect_warmup import write
from examples.locbench.dataset import load_cases
from examples.locbench.env import LocBenchEnv
from examples.locbench.metrics import score
from examples.locbench.repository import prepare


def audit(collection,expected,complete=False):
    from transformers import AutoTokenizer
    verify_harness()
    manifest=json.loads((collection/'manifest.json').read_text())
    if manifest['environment']!=contract():raise ValueError('Environment changed')
    if manifest['split_sha256']!=digest(EXPERIMENT/'data/split.json'):raise ValueError('Split changed')
    for path,sha in manifest['sources'].items():
        if digest(path)!=sha:raise ValueError('Collector source changed')
    source=Path(manifest['source_collection'])
    assert manifest['source_manifest_sha256']==digest(source/'manifest.json')
    split=json.loads((EXPERIMENT/'data/split.json').read_text())
    if not set(manifest['train_ids'])<=set(split['train']) or not set(manifest['validation_ids'])<=set(split['development']):
        raise ValueError('Split membership violation')
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    cases={c.id:c for lane in ['train','development'] for c in load_cases(EXPERIMENT/'data'/f'{lane}.jsonl')}
    rows=0;records=0;maximum=0;seen=set()
    for lane in ['train','validation']:
        for path in (collection/lane).glob('*/sample-*.json'):
            if len(path.suffixes)!=1:continue
            saved=json.loads(path.read_text());q=saved['case_id'];sample=saved['sample']
            assert path==collection/lane/q/f'sample-{sample}.json'
            assert q in manifest['train_ids' if lane=='train' else 'validation_ids']
            assert 0<=sample<manifest['samples_per_question'] and (lane,q,sample) not in seen
            seen.add((lane,q,sample))
            assert saved['source_sha256']==digest(path.with_suffix('.pkl.gz'))
            assert saved['contexts_sha256']==digest(path.with_suffix('.contexts.jsonl.gz'))
            assert saved['manifest_sha256']==digest(collection/'manifest.json')
            for evidence,sha in saved['source_evidence_sha256'].items():
                assert Path(evidence).resolve().is_relative_to(source.resolve())
                assert digest(evidence)==sha
            original=source/lane/q/f'sample-{sample}.json'
            if saved.get('termination')=='model_incomplete_compaction':
                failure=json.loads(original.with_suffix('.failure.json').read_text())
                assert failure['error']=="IncompleteCompactionError('Incomplete compaction reply; refusing to replace task context')"
                assert saved['critic_only'] and saved['rejected_reply_native_tokens_available'] is False
                assert not original.exists()
            else:
                assert saved['source_sha256']==digest(original.with_suffix('.pkl.gz'))
                assert saved['contexts_sha256']==digest(original.with_suffix('.contexts.jsonl.gz'))
            native=pickle.loads(gzip.decompress(path.with_suffix('.pkl.gz').read_bytes()))
            final=native['final'];case=cases[q];metrics=score(final.state.locations,case.gold)
            if saved.get('termination')=='model_incomplete_compaction':
                original_partial=pickle.loads(gzip.decompress(original.with_suffix('.partial.pkl.gz').read_bytes()))
                assert final.turns[:-1]==original_partial['state'].turns
                assert not final.turns[-1].tokens and final.reward==0 and not final.state.submitted
                assert len(native['checkpoints'])==len(original_partial['checkpoints'])
                from dataclasses import fields
                for kept,raw in zip(native['checkpoints'],original_partial['checkpoints'],strict=True):
                    for field in fields(kept):
                        if field.name=='workspace':
                            assert type(kept.workspace) is type(raw.workspace)
                            assert kept.workspace.snapshot()==raw.workspace.snapshot()
                        else:
                            assert getattr(kept,field.name)==getattr(raw,field.name)
                assert native['critic_only'] and not native['rejected_reply_native_tokens_available']
            assert saved['metrics']==metrics and final.reward==metrics['reward'] and final.done
            assert all(t.exact for t in final.turns)
            assert list(final.state.locations)==saved['locations']
            with gzip.open(path.with_suffix('.contexts.jsonl.gz'),'rt') as stream:targets=[json.loads(x) for x in stream]
            assert len(targets)==len(native['checkpoints'])==saved['checkpoints']
            repo=prepare(Path('/weka/scratch/jhu/bvandur1/zjiang31/rollout-controller/data/locbench/repositories'),case.repo,case.base_commit)
            tools=LocBenchEnv(repo,case.gold,call_budget=20,maximum=10).schemas
            for target,snapshot in zip(targets,native['checkpoints'],strict=True):
                assert not snapshot.done and target['group_index']==q and target['target']==metrics['reward']
                assert target['context']==context(snapshot,tools,tokenizer)
                assert target['turn']==snapshot.turns_taken and target['folds']==snapshot.folds
                length=len(tokenizer.encode(target['context'],add_special_tokens=False))+1
                assert length==target['tokens'] and length<=CONTEXT_LIMIT
                maximum=max(maximum,length);rows+=1
            records+=1
    assert records==expected,(records,expected)
    if complete:
        assert records==manifest['samples_per_question']*(len(manifest['train_ids'])+len(manifest['validation_ids']))
        assert json.loads((collection/'collection-complete.json').read_text())['traces']==records
    result=dict(passed=True,traces=records,contexts=rows,max_context_tokens=maximum,
        full_collection=complete,identities_exact=True,manifest_sha256=digest(collection/'manifest.json'))
    write(collection.parent/('collection-audit.json' if complete else 'pilot-audit.json'),result)
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--collection',type=Path,required=True)
    p.add_argument('--expected',type=int,required=True);p.add_argument('--complete',action='store_true')
    args=p.parse_args();audit(args.collection,args.expected,args.complete)
