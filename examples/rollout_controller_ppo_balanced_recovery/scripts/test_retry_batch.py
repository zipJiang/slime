import json
from types import SimpleNamespace

import pytest

from recipe import RECIPE_ID
from retry_batch import restore_batch


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def setup(tmp_path):
    old, new = tmp_path/'failed', tmp_path/'retry'
    source, output = old/'rollouts/train-0042', new/'rollouts/train-0042'
    knobs = dict(hf_checkpoint='base', seed=1, ppo_prior_strength=1,
        ppo_pass_tokens=140000, ppo_search_concurrency=4, rollout_batch_size=1,
        num_critic_only_steps=10, rollout_temperature=1, rollout_top_p=1,
        rollout_top_k=-1, seq_length=32768)
    args = SimpleNamespace(**knobs, ppo_retry_batch_run=str(old),
        start_rollout_id=42, save=str(new/'actor'))
    frozen = dict(policy_version='actor-0032', value_version='critic-0042',
        server_weight_version='1', estimator='direct_branch_td')
    resume = dict(iteration=41, start_rollout_id=42, actor_updates=32,
        critic_updates=42, checkpoints=dict(actor='paired/actor', critic='paired/critic'),
        question_cursor='paired/cursor')
    for root in (old, new):
        write(root/'resume.json', resume)
        write(root/'initial-native-restore-audit.json', dict(passed=True))
        write(root/'critic-publication/critic-0042.json', dict(passed=True,
            publication=dict(manifest=dict(sha256={'weights':'identical'}))))
    write(old/'failed.json', dict(rollout_id=42))
    write(old/'recipe.json', dict(recipe_id=RECIPE_ID, arguments=knobs))
    write(source/'training-lineage.json', dict(collection_round=42, behavior_round=42,
        actor_lag=0, critic_lag=0))
    write(source/'contract.json', dict(**frozen, recipe_id=RECIPE_ID,
        case_keys=['case'], evaluation=False))
    write(source/'target-replay-audit.json', dict(passed=True, scope='whole_batch',
        policy_version=frozen['policy_version'], value_version=frozen['value_version'],
        recipe_id=RECIPE_ID))
    for suffix in ('.json', '.native.pkl.gz', '.actor.jsonl.gz', '.critic.jsonl.gz'):
        (source/f'group-000{suffix}').write_bytes(b'preserved-record')
    return args, frozen, source, output


def test_retry_preserves_records_and_regenerates_training_audits(setup):
    args, frozen, source, output = setup
    report = restore_batch(args, 42, ['case'], frozen, output)
    assert report['collection_reused'] and not report['optimizer_updates_reused']
    assert len(report['source_files']) == 4
    for name in report['source_files']:
        assert (output/name).read_bytes() == (source/name).read_bytes()
    assert not (output/'target-replay-audit.json').exists()
    assert not (output/'training-lineage.json').exists()
    assert restore_batch(args, 43, ['next'], frozen, output) is None


@pytest.mark.parametrize('mutation,match', [
    ('checkpoint', 'same paired checkpoint'), ('cursor', 'cursor'),
    ('stale', 'first batch'), ('audit', 'full target replay'),
    ('weights', 'critic weights'), ('version', 'behavior'),
    ('trained', 'incomplete batch'), ('missing', 'incomplete'),
])
def test_retry_rejects_incompatible_or_incomplete_data(setup, mutation, match):
    args, frozen, source, output = setup
    keys = ['case']
    new = output.parent.parent
    if mutation == 'checkpoint': write(new/'resume.json', dict(iteration=40))
    if mutation == 'cursor': keys = ['different']
    if mutation == 'stale': write(source/'training-lineage.json', dict(collection_round=42,
        behavior_round=41, actor_lag=1, critic_lag=1))
    if mutation == 'audit': write(source/'target-replay-audit.json', dict(passed=False, scope='whole_batch'))
    if mutation == 'weights': write(new/'critic-publication/critic-0042.json', dict(passed=True,
        publication=dict(manifest=dict(sha256={'weights':'different'}))))
    if mutation == 'version': frozen['policy_version'] = 'actor-0033'
    if mutation == 'trained': write(source/'training-complete.json', {})
    if mutation == 'missing': (source/'group-000.actor.jsonl.gz').unlink()
    with pytest.raises(ValueError, match=match):
        restore_batch(args, 42, keys, frozen, output)
    assert not output.exists()
