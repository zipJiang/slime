"""Read audited LocBench returns; reuse the native question-weighted loss format."""
import gzip
import json
from pathlib import Path
import sys
from runtime import EXPERIMENT,digest

sys.path.append(str(EXPERIMENT/'snapshots/native-support-v1'))
from dataset import aggregate


def load(collection):
    collection=Path(collection)
    audit=json.loads((collection.parent/'collection-audit.json').read_text())
    manifest=json.loads((collection/'manifest.json').read_text())
    if not audit['passed'] or not audit['full_collection'] or audit['manifest_sha256']!=digest(collection/'manifest.json'):
        raise ValueError('Full native collection audit required')
    lanes={}
    for lane,key in [('train','train_ids'),('validation','validation_ids')]:
        rows=[]
        for question in manifest[key]:
            for sample in range(manifest['samples_per_question']):
                path=collection/lane/question/f'sample-{sample}.json'
                saved=json.loads(path.read_text())
                contexts=path.with_suffix('.contexts.jsonl.gz')
                if digest(contexts)!=saved['contexts_sha256'] or digest(path.with_suffix('.pkl.gz'))!=saved['source_sha256']:
                    raise ValueError('Warmup source changed after audit')
                with gzip.open(contexts,'rt') as stream:targets=[json.loads(s) for s in stream]
                if any(r['target']!=saved['metrics']['reward'] or r['group_index']!=question for r in targets):
                    raise ValueError('Context/outcome identity mismatch')
                rows.extend(targets)
        lanes[lane]=aggregate(rows)
    return lanes
