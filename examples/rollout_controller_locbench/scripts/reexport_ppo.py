"""Materialize one library cutoff from audited, unchanged prepared evidence."""
import argparse
import gzip
import json
from pathlib import Path
import pickle
import sys
from runtime_v2 import EXPERIMENT,digest
sys.path.append(str(EXPERIMENT/'snapshots/native-support-v1'))
from step_controller.export import to_samples
from targets import split_targets
from collect_ppo import rows,write


def export(source,output,cutoff):
    recipe=json.loads((source/'contract.json').read_text())
    audit=json.loads((source/'target-replay-audit.json').read_text())
    summary=json.loads((source/'summary.json').read_text())
    if not audit['passed'] or audit['contract_sha256']!=digest(source/'contract.json'):
        raise ValueError('Native target readback must precede cutoff materialization')
    output.mkdir(parents=True,exist_ok=False);edges=0;spans=0
    for result in summary['results']:
        group=result['group_index'];path=source/f'group-{group:03d}.prepared.pkl.gz'
        if digest(path)!=result['prepared_sha256']:raise ValueError('Prepared evidence changed')
        with gzip.open(path,'rb') as stream:prepared=pickle.load(stream)
        actor,_=split_targets(to_samples(prepared,group_index=group,min_abs_advantage=cutoff))
        for row in actor:row['metadata'].update(query_id=result['query_id'],policy_version=recipe['policy_version'],
            server_weight_version=recipe['server_weight_version'])
        rows(output/f'group-{group:03d}.actor.jsonl.gz',actor)
        edges+=len({r['metadata']['node_id'] for r in actor});spans+=len(actor)
    result=dict(source=str(source.resolve()),source_contract_sha256=digest(source/'contract.json'),
        source_summary_sha256=digest(source/'summary.json'),min_abs_advantage=cutoff,
        normalization='original question and group_edge_count',actor_edges=edges,actor_spans=spans,
        files={p.name:digest(p) for p in output.glob('*.jsonl.gz')})
    write(output/'export.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--min-abs-advantage',type=float)
    args=p.parse_args();print(json.dumps(export(args.source,args.output,args.min_abs_advantage)),flush=True)
