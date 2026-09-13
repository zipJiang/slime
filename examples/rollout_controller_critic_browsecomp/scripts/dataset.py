"""Question-separated Monte Carlo targets; identical prefixes share observations."""
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import random


def aggregate(rows):
    groups=defaultdict(list)
    for row in rows:
        target=float(row['target'])
        if not math.isfinite(target) or not 0 <= target <= 1:
            raise ValueError('Invalid terminal outcome')
        groups[(row['group_index'],row['context'])].append(row)
    result=[]
    for (question,context), observations in groups.items():
        target=sum(float(r['target']) for r in observations)/len(observations)
        result.append(dict(context=context,target=target,group_index=question,
            turn=observations[0]['turn'],folds=observations[0]['folds'],
            metadata=dict(lane='critic',node_id=hashlib.sha256(context.encode()).hexdigest(),
                target_source='monte_carlo_suffix',diagnostics=dict(
                    observations=len(observations),mean_return=target))))
    return result


def load_dataset(collection, split):
    collection=Path(collection)
    manifest=json.loads((collection/'manifest.json').read_text())
    complete=json.loads((collection/'collection-complete.json').read_text())
    expected=4*(len(manifest['train_ids'])+len(manifest['validation_ids']))
    if complete['traces']!=expected: raise ValueError('Incomplete collection')
    train=set(manifest['train_ids']); validation=set(manifest['validation_ids'])
    if (train & validation or not train <= set(split['train'])
            or not validation <= set(split['development'])
            or (train|validation)&set(split['test'])):
        raise ValueError('Question split violation')
    result={}
    for lane,ids in [('train',manifest['train_ids']),('validation',manifest['validation_ids'])]:
        rows=[]
        for question in ids:
            for sample in range(4):
                path=collection/lane/question/f'sample-{sample}.json'
                summary=json.loads(path.read_text())
                if not summary['done']: raise ValueError('Nonterminal trace has no outcome target')
                if summary['case_id']!=question or summary['sample']!=sample or summary['lane']!=lane:
                    raise ValueError('Trace identity mismatch')
                if hashlib.sha256(path.with_suffix('.pkl.gz').read_bytes()).hexdigest()!=summary['source_sha256']:
                    raise ValueError('Trace checksum mismatch')
                for suffix,key in [('.contexts.jsonl.gz','contexts_sha256'),('.retrieval.json.gz','retrieval_sha256')]:
                    if hashlib.sha256(path.with_suffix(suffix).read_bytes()).hexdigest()!=summary[key]:
                        raise ValueError('Supervision/retrieval checksum mismatch')
                with gzip.open(path.with_suffix('.contexts.jsonl.gz'),'rt') as stream:
                    records=[json.loads(line) for line in stream]
                if len(records)!=summary['checkpoints'] or not records:
                    raise ValueError('Missing checkpoint supervision')
                for record in records:
                    if record['group_index']!=question or record['target']!=float(summary['judge']['correct']):
                        raise ValueError('Checkpoint/outcome mismatch')
                rows.extend(records)
        result[lane]=aggregate(rows)
    return result


def metrics(rows, predictions, baseline):
    if len(rows)!=len(predictions) or not rows: raise ValueError('Missing predictions')
    groups=defaultdict(list)
    for row,prediction in zip(rows,predictions,strict=True):
        if not math.isfinite(prediction) or not 0<=prediction<=1: raise ValueError('Invalid prediction')
        groups[row['group_index']].append((row,prediction))
    def mean(fn):
        return sum(sum(fn(r,p) for r,p in values)/len(values) for values in groups.values())/len(groups)
    # Use the same measure as the loss: every question has equal total mass and
    # its checkpoints divide that mass equally. Otherwise long failed traces
    # would dominate calibration even though they do not dominate optimization.
    bins=[dict(weight=0.,prediction=0.,target=0.,contexts=0,questions=set())
          for _ in range(10)]
    for question,values in groups.items():
        weight=1/(len(groups)*len(values))
        for row,prediction in values:
            bucket=bins[min(9,int(prediction*10))]
            bucket['weight']+=weight
            bucket['prediction']+=weight*prediction
            bucket['target']+=weight*row['target']
            bucket['contexts']+=1
            bucket['questions'].add(question)
    calibration=[]
    for index,bucket in enumerate(bins):
        if not bucket['weight']: continue
        prediction=bucket['prediction']/bucket['weight']
        target=bucket['target']/bucket['weight']
        calibration.append(dict(lower=index/10,upper=(index+1)/10,
            weight=bucket['weight'],contexts=bucket['contexts'],
            questions=len(bucket['questions']),predicted_mean=prediction,
            target_mean=target,absolute_gap=abs(prediction-target)))
    gains=[sum((baseline-r['target'])**2-(p-r['target'])**2 for r,p in values)/len(values)
           for values in groups.values()]
    rng=random.Random(20260912)
    bootstrap=sorted(sum(rng.choices(gains,k=len(gains)))/len(gains) for _ in range(1000))
    return dict(questions=len(groups),checkpoints=len(rows),
        mse=mean(lambda r,p:(p-r['target'])**2),
        mae=mean(lambda r,p:abs(p-r['target'])),
        predicted_mean=mean(lambda r,p:p),target_mean=mean(lambda r,p:r['target']),
        constant_baseline=baseline,baseline_mse=mean(lambda r,p:(baseline-r['target'])**2),
        baseline_mse_improvement_ci95=[bootstrap[24],bootstrap[974]],
        calibration=dict(bins=calibration,
            expected_absolute_gap=sum(b['weight']*b['absolute_gap'] for b in calibration),
            maximum_absolute_gap=max(b['absolute_gap'] for b in calibration)))


def constant_baseline(rows):
    groups=defaultdict(list)
    for row in rows: groups[row['group_index']].append(row['target'])
    return sum(sum(values)/len(values) for values in groups.values())/len(groups)
