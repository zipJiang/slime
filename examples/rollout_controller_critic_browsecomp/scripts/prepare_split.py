"""Reproduce the frozen split without writing benchmark text into public files."""
import argparse
import hashlib
import json
from pathlib import Path
import re

SEED='browsecomp-plus-clean-split-v1-20260912'


def prepare(cases,output):
    rows=[json.loads(line) for line in cases.read_text().splitlines() if line.strip()]
    ids=[str(row['query_id']) for row in rows]
    normalized=[' '.join(row['query'].casefold().split()) for row in rows]
    if len(ids)!=830 or len(set(ids))!=830 or len(set(normalized))!=830:
        raise ValueError('Unexpected inventory or duplicate questions')
    ids.sort(key=lambda key:hashlib.sha256(f'{SEED}/{key}'.encode()).hexdigest())
    split=dict(seed=SEED,cases_sha256=hashlib.sha256(cases.read_bytes()).hexdigest(),
               train=ids[:498],development=ids[498:581],test=ids[581:])
    if output.exists():
        if json.loads(output.read_text())!=split: raise ValueError('Refusing to replace frozen split')
    else:
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(split,indent=2)+'\n')
    words={str(row['query_id']):set(re.findall(r'\w+',row['query'].casefold())) for row in rows}
    overlaps=[]
    for i,a in enumerate(ids):
        for b in ids[i+1:]:
            score=len(words[a]&words[b])/len(words[a]|words[b])
            if score>=.8: overlaps.append(dict(a=a,b=b,jaccard=score))
    report=dict(counts={key:len(split[key]) for key in ['train','development','test']},
                high_overlap_pairs=overlaps)
    output.with_name('split-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    prepare(args.cases,args.output)
