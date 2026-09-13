import math
import pytest
from dataset import aggregate, metrics, constant_baseline
from batches import training_data
from dataset import load_dataset
import gzip
import hashlib
import json


def row(question,context,target):
    return dict(group_index=question,context=context,target=target,turn=0,folds=0)


def test_prefix_means_never_cross_questions():
    rows=aggregate([row('a','root',0),row('a','root',1),row('b','root',0),row('a','fold',1)])
    by_key={(r['group_index'],r['context']):r for r in rows}
    assert len(rows)==3
    assert by_key['a','root']['target']==.5
    assert by_key['a','root']['metadata']['diagnostics']['observations']==2
    assert by_key['b','root']['target']==0


def test_question_weighting_is_independent_of_fold_count():
    rows=aggregate([row('a','root',0),row('a','fold',0),row('b','root',1)])
    assert constant_baseline(rows)==.5
    report=metrics(rows,[0,0,0],.5)
    assert report['mse']==.5
    assert report['baseline_mse']==.25
    fields=[dict(tokens=[1,2],response_length=1,reward=r['target'],loss_mask=[1],
        group_index=r['group_index'],metadata=r['metadata']) for r in rows]
    batch=training_data(fields,lane='critic',expected_groups=['a','b'])
    assert batch['rollout_mask_sums']==[2,2,1]


@pytest.mark.parametrize('target',[math.nan,math.inf,-.1,1.1])
def test_invalid_outcomes_rejected(target):
    with pytest.raises(ValueError): aggregate([row('a','root',target)])


def test_judge_errors_are_not_negative_labels():
    from judge_contract import verdict
    assert verdict('EQUIVALENT','stop') is True
    assert verdict('DIFFERENT','stop') is False
    for text,reason in [('EQUIVALENT','length'),('', 'stop'),('maybe DIFFERENT','stop')]:
        with pytest.raises(ValueError): verdict(text,reason)


def fixture_collection(tmp_path):
    split=dict(train=['a'],development=['b'],test=['c'])
    (tmp_path/'manifest.json').write_text(json.dumps(dict(train_ids=['a'],validation_ids=['b'])))
    (tmp_path/'collection-complete.json').write_text(json.dumps(dict(traces=8)))
    for lane,question in [('train','a'),('validation','b')]:
        for i in range(4):
            path=tmp_path/lane/question/f'sample-{i}.json'
            path.parent.mkdir(parents=True,exist_ok=True)
            payload=row(question,'root '+question,i%2)
            hashes={}
            for suffix,key,raw in [('.pkl.gz','source_sha256',b'local trace'),
                    ('.retrieval.json.gz','retrieval_sha256',gzip.compress(b'{}')),
                    ('.contexts.jsonl.gz','contexts_sha256',gzip.compress((json.dumps(payload)+'\n').encode()))]:
                path.with_suffix(suffix).write_bytes(raw)
                hashes[key]=hashlib.sha256(raw).hexdigest()
            path.write_text(json.dumps(dict(case_id=question,sample=i,lane=lane,
                judge=dict(correct=bool(i%2)),checkpoints=1,done=True,**hashes)))
    return split


def test_complete_data_and_split(tmp_path):
    split=fixture_collection(tmp_path)
    data=load_dataset(tmp_path,split)
    assert len(data['train'])==len(data['validation'])==1
    assert data['train'][0]['target']==.5
    with pytest.raises(ValueError,match='split'):
        load_dataset(tmp_path,dict(train=['a'],development=['c'],test=['b']))


def test_tampered_context_is_rejected(tmp_path):
    split=fixture_collection(tmp_path)
    (tmp_path/'train/a/sample-0.contexts.jsonl.gz').write_bytes(gzip.compress(b'{}\n'))
    with pytest.raises(ValueError,match='checksum'):
        load_dataset(tmp_path,split)
