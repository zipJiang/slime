"""Regrade root evaluations and verify their native traces without using GPUs."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import pickle
import subprocess
import time
from types import SimpleNamespace

from collect_rollouts import EXPERIMENT, HARNESS, corpus_and_split, write_json
from examples.deontic.bench import grade


def audit_trace(final, result, case, policy_version, max_steps=48):
    final.check_continuation()
    assert final.done or final.truncated
    assert final.turns_taken <= max_steps + 1
    assert final.turns_taken == result['task_turns']
    assert final.folds == result['folds']
    assert final.truncated == result['truncated']
    assert final.state.answer == result['answer']
    # A valid submission ends immediately; no later task transition is allowed.
    assert not any(t.done for t in final.transitions[:-1])
    outcome = float(grade(final.state.answer or '', case))
    assert outcome == result['outcome']
    tags = Counter()
    for turn in final.turns:
        if not turn.tokens:
            continue
        assert turn.exact and turn.prefix
        values = turn.logprobs[policy_version]
        assert len(values) == len(turn.tokens)
        assert all(math.isfinite(v) and v <= 1e-5 for v in values)
        tags[turn.tag] += len(turn.tokens)
    cost = dict(input_tokens=sum(len(t.prefix) for t in final.turns if t.tokens),
                output_tokens=sum(len(t.tokens) for t in final.turns),
                generations=sum(bool(t.tokens) for t in final.turns))
    assert cost == result['cost']
    return dict(outcome=outcome, task_turns=final.turns_taken, folds=final.folds,
                truncated=final.truncated, cost=cost, generated_tokens_by_tag=dict(tags))


def audit(directory, branches=8):
    directory = directory.resolve()
    contract = json.loads((directory/'contract.json').read_text())
    summary = json.loads((directory/'summary.json').read_text())
    assert contract['evaluation'] and contract['split'] == 'val'
    assert contract['submissions'] == 1 and contract['max_steps'] == 48
    assert contract['horizon'] == 'shared_task_turns_plus_one_forced_submission'
    assert contract['seed_namespace'] == 'deontic-c1-heldout-20260910'
    for path, key in ((HARNESS/'source-manifest.json', 'harness_manifest_sha256'),
                      (directory/'collector-source.py', 'collector_sha256')):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == contract[key]
    cases, families, split_hash = corpus_and_split(SimpleNamespace(
        questions=directory/'questions.json', split='val', evaluation=True))
    expected = json.loads((EXPERIMENT/'data/question-manifest.json').read_text())['heldout_families']
    assert families == contract['families'] == expected
    assert split_hash == contract['split_sha256']
    for key in ('policy_version', 'server_weight_version', 'value_version'):
        assert summary[key] == contract[key]
    assert summary['critic_requests'] == summary['critic_contexts'] == 0
    assert summary['branches'] == len(families) * branches
    results, traces, hashes = [], [], {}
    for group, family in enumerate(families):
        for branch in range(branches):
            stem = directory/f'group-{group:03d}-branch-{branch}'
            result_path = stem.with_suffix('.json')
            native_path = stem.with_suffix('.native.pkl.gz')
            result_bytes, native_bytes = result_path.read_bytes(), native_path.read_bytes()
            result = json.loads(result_bytes)
            assert (result['group_index'], result['family'], result['branch']) == (group, family, branch)
            final = pickle.loads(gzip.decompress(native_bytes))
            trace = audit_trace(final, result, cases[family], contract['policy_version'])
            traces.append(dict(group_index=group, family=family, branch=branch, **trace))
            results.append(result)
            for path, data in ((result_path, result_bytes), (native_path, native_bytes)):
                hashes[path.name] = hashlib.sha256(data).hexdigest()
    assert summary['results'] == results
    assert summary['correct'] == sum(t['outcome'] for t in traces)
    assert summary['cost'] == {key: sum(t['cost'][key] for t in traces)
                               for key in ('input_tokens', 'output_tokens', 'generations')}
    report = dict(passed=True, scope='whole_root_evaluation', batch=str(directory),
        policy_version=contract['policy_version'], server_weight_version=contract['server_weight_version'],
        branches=len(traces), correct=summary['correct'], cost=summary['cost'],
        mean_task_turns=sum(t['task_turns'] for t in traces)/len(traces),
        mean_folds=sum(t['folds'] for t in traces)/len(traces),
        traces=traces, source_sha256=hashes,
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(directory/'evaluation-audit.json', report)
    return {k: v for k, v in report.items() if k not in ('traces', 'source_sha256')}


def watch(run, branches):
    driver = json.loads((run/'recipe.json').read_text())['driver_step']
    while True:
        for directory in sorted((run/'rollouts').glob('eval-*')):
            if (directory/'summary.json').exists() and not (directory/'evaluation-audit.json').exists():
                print(json.dumps(audit(directory, branches)), flush=True)
        probe = subprocess.run(['squeue', '--steps='+driver, '-h', '-o', '%i'],
                               text=True, capture_output=True)
        if probe.returncode == 0 and driver not in probe.stdout.split():
            print('Driver is terminal; evaluation observation finished.', flush=True)
            return
        time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--batch', type=Path)
    mode.add_argument('--watch-run', type=Path)
    parser.add_argument('--branches', type=int, default=8)
    args = parser.parse_args()
    if args.watch_run:
        watch(args.watch_run.resolve(), args.branches)
    else:
        print(json.dumps(audit(args.batch, args.branches), indent=2))
