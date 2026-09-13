import hashlib
import json

import pytest

from pilot_data import build, select_questions


def fixture():
    split=dict(train=[str(i) for i in range(20)],development=['d0','d1'],test=['t0'])
    manifest=dict(train_ids=['0','1','2'],validation_ids=['d0'])
    return split,manifest


def test_selects_two_disjoint_on_policy_batches():
    split,manifest=fixture()
    result=select_questions(split,manifest,updates=2,batch_size=3)
    assert result['batches']==[['3','4','5'],['6','7','8']]
    assert result['disjoint_from_critic_collection']
    assert result['final_test_excluded']


def test_build_writes_slime_schedule(tmp_path):
    split,manifest=fixture()
    sp=tmp_path/'split.json';mp=tmp_path/'manifest.json';out=tmp_path/'schedule.jsonl'
    sp.write_text(json.dumps(split));mp.write_text(json.dumps(manifest))
    result=build(sp,mp,out,updates=2,batch_size=2)
    rows=[json.loads(line) for line in out.read_text().splitlines()]
    assert [row['metadata']['query_id'] for row in rows]==result['questions']
    assert rows[0]['prompt']==[{'role':'user','content':'3'}]
    assert result['split_sha256']==hashlib.sha256(sp.read_bytes()).hexdigest()


def test_rejects_test_leakage_or_too_few_questions():
    split,manifest=fixture()
    bad=dict(manifest,train_ids=[*manifest['train_ids'],'t0'])
    with pytest.raises(ValueError,match='final-test'):
        select_questions(split,bad,updates=2,batch_size=2)
    with pytest.raises(ValueError,match='Not enough'):
        select_questions(split,manifest,updates=10,batch_size=3)


@pytest.mark.parametrize('updates,batch_size',[(1,6),(2,0)])
def test_requires_real_two_update_protocol(updates,batch_size):
    with pytest.raises(ValueError,match='at least two'):
        select_questions(*fixture(),updates=updates,batch_size=batch_size)
