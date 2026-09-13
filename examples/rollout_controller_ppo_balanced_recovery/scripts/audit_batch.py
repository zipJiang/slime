"""Replay settled controller targets and account for every generated token."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import pickle

# Establish the pinned Python 3.13 harness namespace.
from collect_rollouts import HARNESS, write_json
from balanced_data import metadata_for, validate_batch, split_digest
from step_controller import RefinedTdEstimator, DirectBranchTdEstimator
from recipe import RECIPE_ID, target_source
from step_controller.export import to_samples
from step_controller.preparation import prepare_samples
from step_controller.reward.config import RewardConfig
from targets import split_targets


def rows(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def audit(directory, only_group=None):
    contract = json.loads((directory/'contract.json').read_text())
    summary = json.loads((directory/'summary.json').read_text()) if only_group is None else None
    if contract['evaluation']:
        raise ValueError('This auditor checks search training batches, not evaluation')
    assert contract['recipe_id'] == RECIPE_ID
    validate_batch(contract['case_keys'])
    assert contract['split'] == 'train'
    assert contract['split_sha256'] == split_digest()
    assert contract['case_families'] == [metadata_for(k)['family'] for k in contract['case_keys']]
    assert contract['critic_target_source'] == target_source(contract['estimator'])
    expected_hash = contract['harness_manifest_sha256']
    assert hashlib.sha256((HARNESS/'source-manifest.json').read_bytes()).hexdigest() == expected_hash
    rc = RewardConfig(value_version=contract['value_version'],
        value_prior_strength=contract['value_prior_strength'],
        config_id=f"ppo-kappa-{contract['value_prior_strength']:g}")
    reports = []
    for group, case_key in enumerate(contract['case_keys']):
        if only_group is not None and group != only_group:
            continue
        stem = directory/f'group-{group:03d}'
        with gzip.open(stem.with_suffix('.native.pkl.gz'), 'rb') as stream:
            state = pickle.load(stream)
        if contract.get('horizon') == 'shared_task_turns_plus_one_forced_submission':
            for node in state.nodes.values():
                p = node.payload
                assert p.turns_taken <= contract['max_steps'] + 1, f'{case_key}: horizon exceeded'
                if p.turns_taken > contract['max_steps']:
                    assert p.done or p.truncated, f'{case_key}: live checkpoint past horizon'
        prepared = prepare_samples(state, estimator=(DirectBranchTdEstimator() if contract['estimator'] == 'direct_branch_td' else RefinedTdEstimator()), reward_config=rc,
                                   behavior_version=contract['policy_version'])
        actor, critic = split_targets(to_samples(prepared, group))
        for expected, lane in ((actor, 'actor'), (critic, 'critic')):
            actual = rows(stem.with_suffix(f'.{lane}.jsonl.gz'))
            assert len(actual) == len(expected), f'{case_key}: {lane} record count'
            for a, b in zip(actual, expected, strict=True):
                a = dict(a, metadata=dict(a['metadata']))
                for key, value in dict(case_key=case_key, family=metadata_for(case_key)['family'], policy_version=contract['policy_version'],
                    server_weight_version=contract['server_weight_version']).items():
                    assert a['metadata'].pop(key) == value, f'{case_key}: {key}'
                if lane == 'critic' and contract.get('critic_supervision') == 'nonterminal_only':
                    assert a['metadata'].pop('target_source') == target_source(contract['estimator'])
                    if contract['estimator'] == 'direct_branch_td':
                        assert b['target'] == b['metadata']['diagnostics']['mean_return']
                        assert b['metadata']['diagnostics']['direct_branches'] > 0
                        assert not state.nodes[b['metadata']['node_id']].payload.done
                    boundary = a['metadata'].pop('terminal_boundary')
                    assert type(boundary) is bool and boundary == state.nodes[b['metadata']['node_id']].payload.done
                assert a == b, f'{case_key}: {lane} replay differs at node {b["metadata"]["node_id"]}'
        turns = {id(turn): turn for node in state.nodes.values() for turn in node.payload.turns}
        cost = dict(input_tokens=sum(len(t.prefix) for t in turns.values() if t.tokens),
                    output_tokens=sum(len(t.tokens) for t in turns.values()),
                    generations=sum(bool(t.tokens) for t in turns.values()))
        result = json.loads(stem.with_suffix('.json').read_text())
        assert cost == result['cost'], f'{case_key}: generated token accounting mismatch'
        assert cost['output_tokens'] == sum(sum(r['loss_mask']) for r in actor)
        assert len(result['passes']) == 2
        for key, total in cost.items():
            assert sum(p['cost'][key] for p in result['passes']) == total
        assert not state.stats.get('failures') and not state.stats.get('score_failures')
        tags = Counter()
        for t in turns.values():
            tags[t.tag] += len(t.tokens)
        reports.append(dict(case_key=case_key, actor_spans=len(actor), critic_checkpoints=len(critic),
                            cost=cost, generated_tokens_by_tag=dict(tags)))
    if not reports:
        raise ValueError('No selected groups')
    if summary is not None:
        for key, total in summary['cost'].items():
            assert sum(r['cost'][key] for r in reports) == total
    report = dict(passed=True, batch=str(directory.resolve()), groups=reports,
                  scope='whole_batch' if only_group is None else 'single_group',
                  policy_version=contract['policy_version'], value_version=contract['value_version'],
                  recipe_id=RECIPE_ID, estimator=contract['estimator'])
    name = 'target-replay-audit.json' if only_group is None else f'group-{only_group:03d}.target-replay-audit.json'
    write_json(directory/name, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('batch', type=Path)
    parser.add_argument('--group', type=int)
    args = parser.parse_args()
    print(json.dumps(audit(args.batch, args.group), indent=2))
