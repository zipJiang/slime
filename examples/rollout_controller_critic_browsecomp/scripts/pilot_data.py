"""Deterministic unseen-question schedule for the zero-warmup PPO pilot."""
import hashlib
import json
from pathlib import Path


def select_questions(split, collection_manifest, *, updates=2, batch_size=6):
    if updates < 2 or batch_size < 1:
        raise ValueError('Pilot requires at least two positive joint-update batches')
    split = dict(split)
    manifest = dict(collection_manifest)
    trained = set(manifest['train_ids']) | set(manifest['validation_ids'])
    if trained & set(split['test']):
        raise ValueError('Critic collection includes final-test questions')
    if not trained <= set(split['train']) | set(split['development']):
        raise ValueError('Critic collection falls outside the frozen split')
    eligible = [q for q in split['train'] if q not in trained]
    needed = updates * batch_size
    if len(eligible) < needed:
        raise ValueError('Not enough unseen training questions for the pilot')
    selected = eligible[:needed]
    return dict(schema='browsecomp-zero-warmup-pilot-schedule-v1',
        updates=updates, batch_size=batch_size, questions=selected,
        batches=[selected[i:i+batch_size] for i in range(0, needed, batch_size)],
        critic_collection_questions=sorted(trained),
        disjoint_from_critic_collection=not bool(set(selected) & trained),
        final_test_excluded=not bool(set(selected) & set(split['test'])),
        split_canonical_sha256=hashlib.sha256(json.dumps(split,sort_keys=True,
            separators=(',',':')).encode()).hexdigest())


def write_schedule(path, schedule):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as stream:
        for question in schedule['questions']:
            stream.write(json.dumps(dict(prompt=[dict(role='user',content=question)],
                metadata=dict(query_id=question)))+'\n')
    temporary.replace(path)


def build(split_path, manifest_path, output, *, updates=2, batch_size=6):
    split_path=Path(split_path)
    split=json.loads(split_path.read_text())
    manifest=json.loads(Path(manifest_path).read_text())
    result=select_questions(split,manifest,updates=updates,batch_size=batch_size)
    result['split_sha256']=hashlib.sha256(split_path.read_bytes()).hexdigest()
    write_schedule(output,result)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('split',type=Path)
    parser.add_argument('collection_manifest',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=2)
    parser.add_argument('--batch-size',type=int,default=6)
    args=parser.parse_args()
    result=build(args.split,args.collection_manifest,args.output,
                 updates=args.updates,batch_size=args.batch_size)
    args.audit.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('questions','batches','critic_collection_questions')}))
