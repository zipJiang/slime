import json
from types import SimpleNamespace

import pytest

from grpo_recipe import RECIPE_ID
from resume_grpo import prepare_resume


def saved(tmp_path, recipe_id='locbench-grpo-root8-imitation-v1-20260920'):
    root = tmp_path / 'actor'
    (root / 'rollout').mkdir(parents=True)
    (tmp_path / 'checkpoints').mkdir()
    schedule = tmp_path / 'schedule.jsonl'
    schedule.write_text('question\n')
    args = SimpleNamespace(
        load=str(root), start_rollout_id=None, ckpt_step=None,
        finetune=False, no_load_rng=False, no_load_optim=False,
        rollout_global_dataset=True, global_batch_size=4, rollout_batch_size=4,
        num_steps_per_rollout=1, lr=1e-6, kl_loss_coef=.01,
        rollout_shuffle=False, tensor_model_parallel_size=2,
        grpo_group_size=8, prompt_data=str(schedule),
    )
    (root / 'latest_checkpointed_iteration.txt').write_text('2')
    (root / 'rollout/global_dataset_state_dict_2.pt').touch()
    (root / 'iter_0000002-readback.json').write_text(json.dumps({
        'checkpoint': str(root / 'iter_0000002'), 'role': 'actor',
        'expected_optimizer_steps': 3, 'optimizer_steps': [3],
        'full_storage_read': True, 'finite_tensors': True,
    }))
    (tmp_path / 'recipe.json').write_text(json.dumps({
        'recipe_id': recipe_id, 'arguments': vars(args),
    }))
    (tmp_path / 'checkpoints/round-0002.json').write_text(json.dumps({
        'round_id': 2, 'actor_updates': 3,
    }))
    return args


def test_zero_filter_recipe_can_resume_v1_checkpoint(tmp_path):
    result = prepare_resume(saved(tmp_path))
    assert result['start_rollout_id'] == result['actor_updates'] == 3
    assert result['current_recipe_id'] == RECIPE_ID
    assert result['recipe_change'] == {
        'previous': 'locbench-grpo-root8-imitation-v1-20260920',
        'current': RECIPE_ID,
    }


def test_unknown_recipe_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='explicitly compatible'):
        prepare_resume(saved(tmp_path, 'unrelated-recipe'))
