import hashlib
import json
from pathlib import Path
import pytest
from provenance import dataset_inventory, function_sha256, validate_collection_provenance, validate_source_transition


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path):
    experiment=tmp_path/'experiment'; collection=experiment/'collection'
    source=experiment/'collector.py'; split=experiment/'split.json'
    collection.mkdir(parents=True);source.write_text('pinned\n');split.write_text('{}\n')
    (collection/'infrastructure-manifest.json').write_text('{}\n')
    (collection/'manifest.json').write_text(json.dumps(dict(split_sha256=digest(split),
        sources={'collector.py':digest(source)})))
    return experiment,collection,split,source


def test_training_revalidates_collection_sources_and_split(tmp_path):
    experiment,collection,split,source=fixture(tmp_path)
    report=validate_collection_provenance(experiment,collection,split)
    assert report['sources']['collector.py']==digest(source)
    source.write_text('changed\n')
    with pytest.raises(ValueError,match='source differs'):
        validate_collection_provenance(experiment,collection,split)


def test_training_rejects_changed_split_and_escaping_source(tmp_path):
    experiment,collection,split,_=fixture(tmp_path)
    split.write_text('changed\n')
    with pytest.raises(ValueError,match='split differs'):
        validate_collection_provenance(experiment,collection,split)
    split.write_text('{}\n')
    manifest=json.loads((collection/'manifest.json').read_text())
    outside=tmp_path/'outside.py';outside.write_text('x')
    manifest['sources']={'../outside.py':digest(outside)}
    (collection/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='escapes'):
        validate_collection_provenance(experiment,collection,split)


def test_dataset_inventory_records_exact_target_lineage():
    row=dict(group_index='q',context='visible state',target=.5,turn=0,folds=0,
             metadata=dict(diagnostics=dict(observations=4)))
    inventory=dataset_inventory(dict(train=[row],validation=[dict(row,group_index='v')]))
    assert inventory['train']['questions']==1
    assert inventory['train']['entries'][0]['target']==.5
    assert inventory['train']['entries'][0]['observations']==4
    assert len(inventory['train']['entries'][0]['context_sha256'])==64


def test_function_hash_is_scoped_to_the_context_contract(tmp_path):
    source=tmp_path/'source.py'
    source.write_text('def context(x):\n    return x + 1\n\ndef unrelated():\n    return 2\n')
    first=function_sha256(source,'context')
    source.write_text('def context(x):\n    return x + 1\n\ndef unrelated():\n    return 3\n')
    assert function_sha256(source,'context')==first
    source.write_text('def context(x):\n    return x + 2\n')
    assert function_sha256(source,'context')!=first


def test_recovery_keeps_old_traces_bound_to_old_sources(tmp_path):
    experiment, collection, split, source = fixture(tmp_path)
    recovery = collection/'recovery'
    (recovery/'sources').mkdir(parents=True)
    (recovery/'sources/collector.py').write_bytes(source.read_bytes())
    previous = recovery/'manifest.json'
    previous.write_bytes((collection/'manifest.json').read_bytes())
    row = collection/'train/q/sample-0.json'
    row.parent.mkdir(parents=True)
    row.write_text('{"original":true}')
    inventory = recovery/'retained.json'
    inventory.write_text(json.dumps({'train/q/sample-0.json':digest(row)}))
    manifest = json.loads(previous.read_text())
    manifest['source_transition'] = dict(previous_manifest='recovery/manifest.json',
        previous_manifest_sha256=digest(previous), retained_inventory='recovery/retained.json',
        retained_inventory_sha256=digest(inventory))
    source.write_text('recovered collector\n')
    manifest['sources']['collector.py'] = digest(source)
    (collection/'manifest.json').write_text(json.dumps(manifest))
    validate_collection_provenance(experiment, collection, split)
    newer = row.with_name('sample-1.json')
    newer.write_text('{}')
    with pytest.raises(ValueError, match='resumed collection provenance'):
        validate_source_transition(collection)
    newer.write_text(json.dumps(dict(collection_manifest_sha256=digest(collection/'manifest.json'))))
    validate_source_transition(collection)
    row.write_text('{"original":false}')
    with pytest.raises(ValueError, match='Retained trace changed'):
        validate_source_transition(collection)
