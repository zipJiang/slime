"""Independent frozen-data checks plus a byte-for-byte rebuild in a temporary directory."""
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

root = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
rows = lambda p: [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
manifest = json.loads((root/'manifest.json').read_text())
source = json.loads((root/'source-manifest.json').read_text())
for name, expected in manifest['outputs'].items():
    assert sha(root/name) == expected, name
for record in source['sources'] + source['training_sources']:
    assert sha(Path(record['path'])) == record['sha256'], record['path']
rule = source['family_builder']
assert sha(Path(rule['path'])) == rule['sha256']
spec = importlib.util.spec_from_file_location('historical_family_rules', rule['path'])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
catalog = {}; hard = set()
for record in source['sources']:
    p = Path(record['path'])
    for row in rows(p):
        key = p.parent.name + '/' + row['id']
        if key in catalog:
            assert module.normalized_case(row) == module.normalized_case(catalog[key])
        catalog.setdefault(key, row)
        if p.stem == 'hard': hard.add(key)
families = module.family_map(catalog)
partition = {s: rows(root/f'{s}.jsonl') for s in ['train','test']}
keys = {s: {r['metadata']['case_key'] for r in rs} for s,rs in partition.items()}
assert keys['train'].isdisjoint(keys['test'])
assert keys['train'] | keys['test'] == set(catalog)
assert len(catalog) == 676
family_sets = {s: {families[k] for k in ks} for s,ks in keys.items()}
assert family_sets['train'].isdisjoint(family_sets['test'])
assert len(family_sets['train']) == len(family_sets['test']) == 248
assert not family_sets['test'] & set(manifest['locked_training_families'])
for s, rs in partition.items():
    assert len(rs) == len(keys[s]) == 338
    cell_mass = defaultdict(float)
    for r in rs:
        m = r['metadata']; k = m['case_key']
        assert r['prompt'] == [{'role':'user','content':k}]
        assert m['family'] == families[k] and m['hard'] == (k in hard)
        cell_mass[m['domain'],m['hard']] += m['sampling_weight']
    assert len(cell_mass) == 6
    assert all(math.isclose(w,1/6) for w in cell_mass.values())
    for d in ['airline','sara_numeric','sara_binary']:
        path = root/'corpus'/d/f'{s}.jsonl'
        records = rows(path)
        assert {d+'/'+r['id'] for r in records} == {k for k in keys[s] if k.startswith(d+'/')}
        for r in records:
            assert r == catalog[d+'/'+r['id']]
            assert all((path.parent/p).is_file() for p in r['statutes'])
for original, expected in manifest['statute_sources'].items():
    assert sha(Path(original)) == expected
schedule_path = root/'train-schedule-6000-seed20260912.jsonl'
schedule = rows(schedule_path)
assert len(schedule) == 6000
assert {r['metadata']['case_key'] for r in schedule} <= keys['train']
assert Counter((r['metadata']['domain'],r['metadata']['hard']) for r in schedule) == {
    (d,h):1000 for d in ['airline','sara_numeric','sara_binary'] for h in [False,True]}
for i in range(0,len(schedule),6):
    assert len({(r['metadata']['domain'],r['metadata']['hard']) for r in schedule[i:i+6]}) == 6
with tempfile.TemporaryDirectory(prefix='deontic-split-audit-') as tmp:
    rebuilt = Path(tmp)/'rebuilt'
    subprocess.run([sys.executable,str(root/'build.py'),'build','--source-manifest',
        str(root/'source-manifest.json'),'--output',str(rebuilt), '--test-fraction',
        str(manifest['test_fraction']),'--seed',str(manifest['seed'])],check=True,capture_output=True)
    for name, expected in manifest['outputs'].items():
        assert sha(rebuilt/name) == expected, name
    assert (rebuilt/'manifest.json').read_bytes() == (root/'manifest.json').read_bytes()
report = dict(passed=True, cases=676, families=496, train_cases=338,test_cases=338,
    train_families=248,test_families=248, leakage_checks='passed',
    historical_family_rule_equivalence=True, byte_identical_rebuild=True,
    balanced_schedule_rows=6000, all_six_cells_have_1000_draws=True,
    manifest_sha256=sha(root/'manifest.json'), audit_script_sha256=sha(Path(__file__)),
    schedule_sha256=sha(schedule_path))
(root/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
