"""Validate immutable collection inputs again at the training boundary."""
import ast
import hashlib
import json
from pathlib import Path


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def function_sha256(path, name):
    source=Path(path).read_text()
    tree=ast.parse(source)
    matches=[node for node in tree.body
             if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name==name]
    if len(matches)!=1:
        raise ValueError(f'Expected exactly one top-level function named {name}')
    segment=ast.get_source_segment(source,matches[0])
    if not segment:
        raise ValueError(f'Could not recover source for function {name}')
    return hashlib.sha256(segment.encode()).hexdigest()


def validate_source_transition(collection):
    """Keep pre-recovery traces tied to their archived collection protocol."""
    collection = Path(collection).resolve()
    manifest = json.loads((collection/'manifest.json').read_text())
    transition = manifest.get('source_transition')
    if transition is None:
        return

    def inside(relative):
        path = (collection/relative).resolve()
        if not path.is_relative_to(collection):
            raise ValueError(f'Recovery path escapes collection: {relative}')
        return path

    previous = inside(transition['previous_manifest'])
    inventory = inside(transition['retained_inventory'])
    if sha256(previous) != transition['previous_manifest_sha256'] or sha256(inventory) != transition['retained_inventory_sha256']:
        raise ValueError('Recovery provenance checksum mismatch')
    old = json.loads(previous.read_text())
    if old['split_sha256'] != manifest['split_sha256']:
        raise ValueError('Recovery changed the question split')
    for relative, expected in old['sources'].items():
        archived = inside(str(previous.parent.relative_to(collection)/'sources'/relative))
        if sha256(archived) != expected:
            raise ValueError(f'Archived collection source differs: {relative}')
    retained = json.loads(inventory.read_text())
    for relative, expected in retained.items():
        if sha256(inside(relative)) != expected:
            raise ValueError(f'Retained trace changed after recovery: {relative}')
    current = sha256(collection/'manifest.json')
    for lane in ['train', 'validation']:
        for path in (collection/lane).glob('*/sample-*.json'):
            if path.name.endswith('.failure.json') or str(path.relative_to(collection)) in retained:
                continue
            if json.loads(path.read_text()).get('collection_manifest_sha256') != current:
                raise ValueError(f'Trace lacks the resumed collection provenance: {path}')


def validate_collection_provenance(experiment, collection, split_path):
    experiment=Path(experiment).resolve()
    collection=Path(collection).resolve()
    split_path=Path(split_path).resolve()
    manifest=json.loads((collection/'manifest.json').read_text())
    if sha256(split_path) != manifest['split_sha256']:
        raise ValueError('Frozen question split differs from the collection manifest')
    checked={}
    for relative,expected in manifest['sources'].items():
        source=(experiment/relative).resolve()
        if not source.is_relative_to(experiment):
            raise ValueError(f'Collection source escapes experiment root: {relative}')
        actual=sha256(source)
        if actual != expected:
            raise ValueError(f'Collection source differs from manifest: {relative}')
        checked[relative]=actual
    infrastructure=collection/'infrastructure-manifest.json'
    if not infrastructure.is_file():
        raise ValueError('Missing collection infrastructure manifest')
    validate_source_transition(collection)
    return dict(split_sha256=manifest['split_sha256'],sources=checked,
                manifest_sha256=sha256(collection/'manifest.json'),
                infrastructure_manifest_sha256=sha256(infrastructure))


def dataset_inventory(data):
    result={}
    for lane,rows in data.items():
        entries=[]
        for row in rows:
            diagnostics=row['metadata']['diagnostics']
            entries.append(dict(question=row['group_index'],
                context_sha256=hashlib.sha256(row['context'].encode()).hexdigest(),
                target=row['target'],observations=diagnostics['observations'],
                turn=row['turn'],folds=row['folds']))
        result[lane]=dict(questions=len({r['group_index'] for r in rows}),
                          contexts=len(rows),entries=entries)
    return result
