"""Read back trace snapshots and verify the exact critic conditioning strings."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import sys

EXPERIMENT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(EXPERIMENT/'snapshots/harness'))
from collect import context, write
from provenance import validate_source_transition
from examples.browsercomp_plus.env import RetrievalArchive, SearchEnv
from transformers import AutoTokenizer


def audit(root):
    manifest=json.loads((root/'manifest.json').read_text())
    for name,sha in manifest['sources'].items():
        assert hashlib.sha256((EXPERIMENT/name).read_bytes()).hexdigest()==sha, name
    validate_source_transition(root)
    tokenizer=AutoTokenizer.from_pretrained(manifest['actor'],local_files_only=True)
    tools=SearchEnv(RetrievalArchive()).schemas
    expected={(lane,case_id,sample)
              for lane,key in [('train','train_ids'),('validation','validation_ids')]
              for case_id in manifest[key]
              for sample in range(manifest['samples_per_question'])}
    observed=set()
    count=checkpoints=folds=successes=0
    max_tokens=0
    for lane,key in [('train','train_ids'),('validation','validation_ids')]:
        for path in sorted((root/lane).glob('*/sample-*.json')):
            if path.name.endswith('.failure.json'): continue
            row=json.loads(path.read_text())
            try: sample=int(path.stem.removeprefix('sample-'))
            except ValueError as exc: raise ValueError(f'Invalid trace filename: {path}') from exc
            identity=(lane,path.parent.name,sample)
            if (identity not in expected or row['case_id'] != path.parent.name
                    or row['sample'] != sample or row['lane'] != lane):
                raise ValueError(f'Trace identity mismatch: {path}')
            if identity in observed: raise ValueError(f'Duplicate trace identity: {identity}')
            observed.add(identity)
            for suffix,sha in [('.pkl.gz','source_sha256'),('.contexts.jsonl.gz','contexts_sha256'),
                               ('.retrieval.json.gz','retrieval_sha256')]:
                assert hashlib.sha256(path.with_suffix(suffix).read_bytes()).hexdigest()==row[sha]
            trace=pickle.loads(gzip.decompress(path.with_suffix('.pkl.gz').read_bytes()))
            with gzip.open(path.with_suffix('.contexts.jsonl.gz'),'rt') as stream:
                records=[json.loads(line) for line in stream]
            assert len(records)==len(trace['checkpoints'])==row['checkpoints']
            assert trace['final'].done
            assert trace['final'].truncated==row['horizon_finished']
            for record,snapshot in zip(records,trace['checkpoints'],strict=True):
                assert not snapshot.done and not snapshot.truncated
                assert snapshot.state.answer is None
                assert context(snapshot,tools,tokenizer)==record['context']
                assert record['target']==float(row['judge']['correct'])
                tokens=len(tokenizer.encode(record['context'],add_special_tokens=False))
                assert 0<tokens<32768
                max_tokens=max(max_tokens,tokens)
            count+=1;checkpoints+=len(records);folds+=row['folds'];successes+=bool(row['judge']['correct'])
    if count==0: raise ValueError('No completed traces to audit')
    complete=root/'collection-complete.json'
    full_collection=False
    if complete.exists():
        marker=json.loads(complete.read_text())
        if marker.get('traces') != len(expected) or observed != expected:
            missing=sorted(expected-observed)
            extra=sorted(observed-expected)
            raise ValueError(f'Complete marker does not match trace inventory: missing={missing[:5]}, extra={extra[:5]}')
        full_collection=True
    return dict(passed=True,traces=count,expected_traces=len(expected),
        full_collection=full_collection,identities_exact=observed==expected,
        checkpoints=checkpoints,folds=folds,
        successes=successes,max_context_tokens=max_tokens,exact_snapshot_context_readback=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('collection',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.collection)
    write(args.output,result)
    print(json.dumps(result))
