"""Re-export audited historical trees for validation only; never training replay."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import shutil

from collect_rollouts import HARNESS, write_json, write_rows
from step_controller import DirectBranchTdEstimator, RefinedTdEstimator
from step_controller.export import to_samples
from step_controller.preparation import prepare_samples
from step_controller.reward.config import RewardConfig
from audit_batch import audit
from recipe import RECIPE_ID, target_source
from targets import split_targets


def replay(source, output, estimator_name):
    if output.exists():
        raise ValueError('Use a fresh validation output directory')
    original_audit = json.loads((source/'target-replay-audit.json').read_text())
    if not original_audit['passed'] or original_audit['scope'] != 'whole_batch':
        raise ValueError('Source needs a complete original target audit')
    output.mkdir(parents=True)
    contract = json.loads((source/'contract.json').read_text())
    contract.update(recipe_id=RECIPE_ID, estimator=estimator_name,
        critic_target_source=target_source(estimator_name), validation_only=True,
        source_batch=str(source.resolve()),
        source_contract_sha256=hashlib.sha256((source/'contract.json').read_bytes()).hexdigest(),
        harness_manifest_sha256=hashlib.sha256((HARNESS/'source-manifest.json').read_bytes()).hexdigest())
    write_json(output/'contract.json', contract)
    rc = RewardConfig(value_version=contract['value_version'],
        value_prior_strength=contract['value_prior_strength'],
        config_id=f"ppo-kappa-{contract['value_prior_strength']:g}")
    estimator = DirectBranchTdEstimator() if estimator_name == 'direct_branch_td' else RefinedTdEstimator()
    results = []
    native_hashes = {}
    for group, family in enumerate(contract['families']):
        stem = f'group-{group:03d}'
        native = source/f'{stem}.native.pkl.gz'
        native_hashes[native.name] = hashlib.sha256(native.read_bytes()).hexdigest()
        with gzip.open(native, 'rb') as stream:
            tree = pickle.load(stream)
        before = pickle.dumps(tree)
        batch = prepare_samples(tree, estimator=estimator, reward_config=rc,
                                behavior_version=contract['policy_version'])
        actor, critic = split_targets(to_samples(batch, group))
        assert pickle.dumps(tree) == before
        for rows, lane in ((actor, 'actor'), (critic, 'critic')):
            for row in rows:
                row['metadata'].update(family=family, policy_version=contract['policy_version'],
                    server_weight_version=contract['server_weight_version'])
                if lane == 'critic':
                    row['metadata'].update(target_source=target_source(estimator_name),
                        terminal_boundary=tree.nodes[row['metadata']['node_id']].payload.done)
            write_rows(output/f'{stem}.{lane}.jsonl.gz', rows)
        result = json.loads((source/f'{stem}.json').read_text())
        result.update(actor_spans=len(actor), critic_checkpoints=len(critic),
                      actor_tokens=sum(sum(r['loss_mask']) for r in actor))
        write_json(output/f'{stem}.json', result)
        results.append(result)
        shutil.copyfile(native, output/native.name)
    summary = json.loads((source/'summary.json').read_text())
    summary['results'] = results
    write_json(output/'summary.json', summary)
    report = audit(output)
    report.update(validation_only=True, source_native_sha256=native_hashes,
                  source_batch=str(source.resolve()), trees_unchanged=True)
    write_json(output/'preflight-report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--estimator', choices=['refined_td', 'direct_branch_td'], required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.source, args.output, args.estimator), indent=2))
