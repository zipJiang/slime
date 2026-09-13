import json
from audit_sampling import audit, seed


def test_seed_is_reproducible_and_separates_samples():
    assert seed('train','q',0,0)==seed('train','q',0,0)
    assert len({seed('train','q',sample,0) for sample in range(4)})==4
    assert seed('train','q',0,0)!=seed('train','q',0,1)


def test_manifest_audit_covers_every_planned_trace(tmp_path):
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(samples_per_question=4,
        train_ids=['a','b'],validation_ids=['c'])))
    report=audit(manifest,max_calls=8)
    assert report['questions']==3 and report['traces']==12
    assert report['distinct_within_question_at_every_call']
