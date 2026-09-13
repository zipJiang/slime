"""Audit deterministic, independent sampling streams for critic collection."""
import argparse
import hashlib
import json
from pathlib import Path


NAMESPACE='browsecomp-critic-v1'


def seed(lane, question, sample, call):
    key=f'{NAMESPACE}/{lane}/{question}/{sample}/{call}'
    return int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%(2**31)


def audit(manifest, max_calls=64):
    manifest=json.loads(Path(manifest).read_text())
    samples=manifest['samples_per_question']
    if samples<2 or max_calls<1:
        raise ValueError('Sampling audit needs multiple samples and at least one call')
    streams=[]
    for lane,key in [('train','train_ids'),('validation','validation_ids')]:
        for question in manifest[key]:
            per_sample=[[seed(lane,question,sample,call) for call in range(max_calls)]
                        for sample in range(samples)]
            for call in range(max_calls):
                values=[stream[call] for stream in per_sample]
                if len(set(values))!=samples:
                    raise ValueError(f'Sampling seed collision within question {question} at call {call}')
            streams.append(dict(lane=lane,question=question,
                stream_sha256=[hashlib.sha256(json.dumps(values).encode()).hexdigest()
                               for values in per_sample],root_seeds=[values[0] for values in per_sample]))
    return dict(passed=True,namespace=NAMESPACE,questions=len(streams),samples_per_question=samples,
        audited_calls_per_trace=max_calls,traces=len(streams)*samples,
        distinct_within_question_at_every_call=True,streams=streams)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-calls',type=int,default=64)
    args=parser.parse_args()
    result=audit(args.manifest,args.max_calls)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='streams'}))
