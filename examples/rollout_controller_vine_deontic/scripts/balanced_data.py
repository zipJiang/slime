"""Frozen case identity, family separation, and six-cell sampling contracts."""
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[1]
BUNDLE = EXPERIMENT / 'data/split'
EVAL_SEED = 'deontic-balanced-test-v1-20260912'
DOMAINS = ('airline', 'sara_numeric', 'sara_binary')


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def split_digest():
    return hashlib.sha256((BUNDLE / 'manifest.json').read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def inventories():
    manifest = json.loads((BUNDLE / 'manifest.json').read_text())
    for name, expected in manifest['outputs'].items():
        if hashlib.sha256((BUNDLE / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen split hash mismatch: {name}')
    parts = {part: {r['metadata']['case_key']: r['metadata']
                    for r in read_rows(BUNDLE / f'{part}.jsonl')}
             for part in ('train', 'test')}
    if any(len(v) != 338 for v in parts.values()):
        raise ValueError('Expected 338 unique cases per frozen partition')
    if set(parts['train']) & set(parts['test']):
        raise ValueError('Case leakage between train and test')
    families = {p: {m['family'] for m in rs.values()} for p, rs in parts.items()}
    if families['train'] & families['test']:
        raise ValueError('Family leakage between train and test')
    return parts


def metadata_for(case_key):
    parts = inventories()
    return parts['train'].get(case_key) or parts['test'][case_key]


def test_case_keys():
    return list(inventories()['test'])


def validate_batch(case_keys):
    """Repeated cases are distinct roots; every six roots must cover all cells."""
    train = inventories()['train']
    if not case_keys or len(case_keys) % 6 or any(k not in train for k in case_keys):
        raise ValueError('Training requires complete six-cell batches from the train allowlist')
    expected = Counter({(d, hard): 1 for d in DOMAINS for hard in (False, True)})
    for i in range(0, len(case_keys), 6):
        observed = Counter((train[k]['domain'], train[k]['hard']) for k in case_keys[i:i+6])
        if observed != expected:
            raise ValueError('Training schedule lost its six-cell balance')


def validate_selection(selected, split, evaluation):
    part = 'test' if split == 'val' else split
    if part not in ('train', 'test'):
        raise ValueError('Unknown split')
    if part == 'test' and not evaluation:
        raise ValueError('Test data may only be used for root evaluation')
    allowed = inventories()[part]
    if not selected or any(k not in allowed for k in selected):
        raise ValueError('Case missing or outside authorized split')
    if evaluation and len(set(selected)) != len(selected):
        raise ValueError('Evaluation cases must be unique')
    return part


def corpus_and_split(args):
    selected = json.loads(args.questions.read_text())
    part = validate_selection(selected, args.split, args.evaluation)
    allowed = inventories()[part]
    from examples.deontic.bench import load_cases
    cases = {f'{domain}/{case.id}': case for domain in DOMAINS
             for case in load_cases(domain, part, BUNDLE / 'corpus', None)}
    if set(cases) != set(allowed):
        raise ValueError('Frozen corpus and question inventory disagree')
    return cases, selected, split_digest()


def request_seed(namespace, group, case_key, branch, draw):
    key = f'{namespace}/{group}/{case_key}/{branch}/{draw}'
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 2**31
