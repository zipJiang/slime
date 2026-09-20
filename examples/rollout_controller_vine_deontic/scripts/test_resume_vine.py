import json
from types import SimpleNamespace

import pytest

from recipe import RECIPE_ID
from resume_vine import prepare_resume, validate_actor_restore


@pytest.fixture
def saved(tmp_path):
    root = tmp_path/'actor'
    root.mkdir()
    (root/'rollout').mkdir()
    (tmp_path/'checkpoints').mkdir()
    data = tmp_path/'schedule.jsonl'
    data.write_text('question\n')
    args = SimpleNamespace(load=str(root), start_rollout_id=None, ckpt_step=None,
        finetune=False, no_load_rng=False, no_load_optim=False,
        rollout_global_dataset=True, global_batch_size=6, rollout_batch_size=6,
        num_steps_per_rollout=1, vine_group_size=16, lr=1e-6, kl_loss_coef=.01,
        rollout_shuffle=False, tensor_model_parallel_size=2, prompt_data=str(data))
    (root/'latest_checkpointed_iteration.txt').write_text('11')
    (root/'rollout/global_dataset_state_dict_11.pt').touch()
    (root/'iter_0000011-readback.json').write_text(json.dumps(dict(
        checkpoint=str(root/'iter_0000011'), role='actor', expected_optimizer_steps=12,
        optimizer_steps=[12], full_storage_read=True, finite_tensors=True)))
    (tmp_path/'recipe.json').write_text(json.dumps(dict(recipe_id=RECIPE_ID, arguments=vars(args))))
    (tmp_path/'checkpoints/round-0011.json').write_text(json.dumps(dict(round_id=11, actor_updates=12)))
    return args, root


def test_resume_preserves_completed_update_count(saved):
    args, _ = saved
    result = prepare_resume(args)
    assert result['actor_updates'] == args.start_rollout_id == 12
    assert result['source_recipe_id'] == RECIPE_ID
    assert result['current_recipe_id'] == RECIPE_ID
    assert result['recipe_change'] is None


def test_resume_records_explicit_v2_to_v3_recipe_migration(saved):
    args, root = saved
    recipe_path = root.parent/'recipe.json'
    recipe = json.loads(recipe_path.read_text())
    source = 'vine-ppo-balanced-k1-base-both-null-g5-v2-20260919'
    recipe['recipe_id'] = source
    recipe_path.write_text(json.dumps(recipe))
    result = prepare_resume(args)
    assert result['source_recipe_id'] == source
    assert result['current_recipe_id'] == RECIPE_ID
    assert result['recipe_change'] == dict(previous=source, current=RECIPE_ID)


def test_resume_rejects_unlisted_recipe(saved):
    args, root = saved
    recipe_path = root.parent/'recipe.json'
    recipe = json.loads(recipe_path.read_text())
    recipe['recipe_id'] = 'unrelated-experiment'
    recipe_path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match='explicitly compatible'):
        prepare_resume(args)


@pytest.mark.parametrize('flag', ['no_load_optim', 'no_load_rng', 'finetune'])
def test_resume_rejects_reset_flags(saved, flag):
    args, _ = saved
    setattr(args, flag, True)
    with pytest.raises(ValueError, match='retain optimizer'):
        prepare_resume(args)


def test_resume_rejects_missing_cursor(saved):
    args, root = saved
    (root/'rollout/global_dataset_state_dict_11.pt').unlink()
    with pytest.raises(ValueError, match='question cursor'):
        prepare_resume(args)


def test_resume_rejects_partial_audit(saved):
    args, root = saved
    path = root/'iter_0000011-readback.json'
    audit = json.loads(path.read_text()); audit['full_storage_read'] = False
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match='readback'):
        prepare_resume(args)


def test_resume_rejects_new_schedule(saved, tmp_path):
    args, _ = saved
    other = tmp_path/'new.jsonl'; other.write_text('different question\n')
    args.prompt_data = str(other)
    with pytest.raises(ValueError, match='schedule'):
        prepare_resume(args)


def test_resume_rejects_changed_group_size(saved):
    args, _ = saved; args.vine_group_size = 8
    with pytest.raises(ValueError, match='vine_group_size'):
        prepare_resume(args)


def test_native_optimizer_audit_rejects_wrong_step():
    good = dict(steps=[12], states=188, scheduler_samples=72,
        no_load_optim=False, no_load_rng=False, finetune=False)
    reports = [dict(good) for _ in range(4)]
    assert validate_actor_restore(dict(actor_updates=12), reports, 6, 4)['passed']
    reports[2]['steps'] = [0]
    with pytest.raises(ValueError, match='rank 2'):
        validate_actor_restore(dict(actor_updates=12), reports, 6, 4)


def test_resume_records_explicit_group_size_transition(saved):
    args, _ = saved
    args.vine_group_size = 5
    args.resume_vine_group_size_from = 16
    result = prepare_resume(args)
    assert result['group_size_change'] == dict(previous=16, current=5)
    assert result['actor_updates'] == args.start_rollout_id == 12


def test_resume_rejects_wrong_authorized_source_group_size(saved):
    args, _ = saved
    args.vine_group_size = 5
    args.resume_vine_group_size_from = 8
    with pytest.raises(ValueError, match='source vine_group_size'):
        prepare_resume(args)


def test_resume_records_explicit_value_rollout_transition(saved):
    args, _ = saved
    args.vine_value_rollouts_per_state = 3
    args.resume_vine_value_rollouts_from = 1
    result = prepare_resume(args)
    assert result['value_rollout_change'] == dict(previous=1, current=3)
