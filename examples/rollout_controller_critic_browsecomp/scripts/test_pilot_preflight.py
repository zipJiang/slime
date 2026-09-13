import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from pilot_preflight import build
from pilot_data import select_questions


def inputs(tmp_path):
    base = tmp_path/'base'; base.mkdir()
    checkpoint = tmp_path/'critic'; (checkpoint/'iter_0000015').mkdir(parents=True)
    context = tmp_path/'collect.py'; context.write_text('def context(): pass\n')
    cases = tmp_path/'cases.jsonl'; cases.write_text('{}\n')
    retriever = tmp_path/'retriever'; retriever.mkdir()
    split=tmp_path/'split.json';split.write_text(json.dumps(dict(
        train=[str(i) for i in range(12)],development=['dev'],test=['test'])))
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(
        dict(train_ids=[],validation_ids=[])))
    schedule_value=select_questions(json.loads(split.read_text()),json.loads(manifest.read_text()))
    schedule_value['split_sha256']=hashlib.sha256(split.read_bytes()).hexdigest()
    schedule = tmp_path/'schedule.json';schedule.write_text(json.dumps(schedule_value))
    candidate_path = tmp_path/'candidate.json'; candidate_path.write_text('{}')
    candidate = dict(base_actor=str(base), critic=dict(load=str(checkpoint), ckpt_step=15))
    kwargs=dict(candidate_path=candidate_path,context_source=context,
        schedule_audit=schedule,cases=cases,retriever_code=retriever,
        base_actor=base,updates=2,batch_size=6,critic_only_steps=0,
        train_gpus=4,rollout_gpus=2,critic_replica_host='10.0.0.2',
        retriever_url='http://retriever:8000',judge_url='http://judge:30000',
        split_path=split,collection_manifest=manifest)
    return candidate, kwargs


def test_build_records_exact_resource_and_data_contract(tmp_path):
    candidate, kwargs = inputs(tmp_path)
    with patch('pilot_preflight.require_pilot_candidate',return_value=candidate), \
         patch('pilot_preflight.zero_warmup_role_arguments',return_value=(None,None,{'mode':'pilot'})):
        result = build(**kwargs)
    assert result['passed'] and result['total_gpus'] == 9
    assert result['schedule_batches'][1][-1] == '11'
    assert result['lineage'] == {'mode':'pilot'}


def test_two_training_gpus_preserve_question_and_update_contract(tmp_path):
    candidate,kwargs=inputs(tmp_path)
    with patch('pilot_preflight.require_pilot_candidate',return_value=candidate), \
         patch('pilot_preflight.zero_warmup_role_arguments',return_value=(None,None,{})):
        large=build(**kwargs)
        small=build(**dict(kwargs,train_gpus=2))
    assert small['total_gpus']==7
    for key in ['schedule_batches','batch_size','updates','critic_only_steps',
                'candidate_sha256','context_source_sha256']:
        assert small[key]==large[key]


@pytest.mark.parametrize('field,value',[
    ('updates',3),('batch_size',5),('critic_only_steps',1),
    ('train_gpus',3),('rollout_gpus',3),('judge_url','not-a-url')])
def test_build_rejects_contract_drift(tmp_path,field,value):
    candidate, kwargs = inputs(tmp_path); kwargs[field] = value
    with patch('pilot_preflight.require_pilot_candidate',return_value=candidate), \
         patch('pilot_preflight.zero_warmup_role_arguments',return_value=(None,None,{})), \
         pytest.raises(ValueError):
        build(**kwargs)


def test_build_rejects_duplicate_questions(tmp_path):
    candidate, kwargs = inputs(tmp_path)
    schedule = json.loads(Path(kwargs['schedule_audit']).read_text())
    schedule['batches'][1][0] = schedule['batches'][0][0]
    Path(kwargs['schedule_audit']).write_text(json.dumps(schedule))
    with patch('pilot_preflight.require_pilot_candidate',return_value=candidate), \
         patch('pilot_preflight.zero_warmup_role_arguments',return_value=(None,None,{})), \
         pytest.raises(ValueError,match='repeats'):
        build(**kwargs)


def test_build_regenerates_selection_instead_of_trusting_flags(tmp_path):
    candidate, kwargs = inputs(tmp_path)
    schedule = json.loads(Path(kwargs['schedule_audit']).read_text())
    schedule['questions'][-1] = schedule['batches'][-1][-1] = 'unseen-but-unfrozen'
    Path(kwargs['schedule_audit']).write_text(json.dumps(schedule))
    with patch('pilot_preflight.require_pilot_candidate',return_value=candidate), \
         patch('pilot_preflight.zero_warmup_role_arguments',return_value=(None,None,{})), \
         pytest.raises(ValueError,match='regenerated'):
        build(**kwargs)
