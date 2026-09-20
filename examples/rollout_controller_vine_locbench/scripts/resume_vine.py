"""Require a fully audited actor checkpoint and matching dataset cursor."""
import json
from pathlib import Path

from recipe import RECIPE_ID


def prepare_resume(args):
    root = Path(args.load)
    tracker = root / 'latest_checkpointed_iteration.txt'
    if not tracker.exists():
        if args.start_rollout_id not in (None, 0):
            raise ValueError('Fresh HF initialization starts at round zero')
        return None
    recipe = json.loads((root.parent / 'recipe.json').read_text())
    if recipe.get('recipe_id') != RECIPE_ID:
        raise ValueError('Resume requires the same VinePPO recipe')
    if (root.parent / 'scientific-rejection.json').exists():
        raise ValueError('Cannot resume a scientifically rejected run')
    iteration = int(tracker.read_text().strip())
    if iteration < 0:
        raise ValueError('Invalid checkpoint iteration')
    start = iteration + 1
    if args.start_rollout_id not in (None, start) or args.ckpt_step not in (None, iteration):
        raise ValueError('Resume cursor disagrees with checkpoint')
    if args.finetune or args.no_load_optim or args.no_load_rng:
        raise ValueError('Resume must retain optimizer, scheduler, and RNG history')
    checkpoint = root / f'iter_{iteration:07d}'
    audit = json.loads((root / f'iter_{iteration:07d}-readback.json').read_text())
    if (Path(audit['checkpoint']).resolve() != checkpoint.resolve()
            or audit['role'] != 'actor' or audit['expected_optimizer_steps'] != start
            or audit['optimizer_steps'] != [start] or not audit['full_storage_read']
            or not audit['finite_tensors']):
        raise ValueError('Checkpoint requires a matching full tensor readback audit')
    cursor = root / 'rollout' / f'global_dataset_state_dict_{iteration}.pt'
    if not args.rollout_global_dataset or not cursor.is_file():
        raise ValueError('Resume requires the saved question cursor')
    saved = json.loads((root.parent / 'checkpoints' / f'round-{iteration:04d}.json').read_text())
    if saved['actor_updates'] != start or saved['round_id'] != iteration:
        raise ValueError('Checkpoint was not committed at the matching training boundary')
    # A resume is a continuation, so retain the training schedule and loss recipe.
    previous = recipe['arguments']
    for key in ('global_batch_size', 'rollout_batch_size', 'num_steps_per_rollout',
                'lr', 'kl_loss_coef', 'rollout_shuffle',
                'tensor_model_parallel_size'):
        if previous[key] != getattr(args, key):
            raise ValueError(f'Resume changes training setting: {key}')
    group_change = None
    authorized_previous = getattr(args, 'resume_vine_group_size_from', None)
    if authorized_previous is not None and authorized_previous != previous['vine_group_size']:
        raise ValueError('Authorized source vine_group_size does not match checkpoint')
    if previous['vine_group_size'] != args.vine_group_size:
        if authorized_previous is None or args.vine_group_size < 1:
            raise ValueError('Resume changes training setting: vine_group_size')
        group_change = dict(previous=previous['vine_group_size'], current=args.vine_group_size)
    if Path(previous['prompt_data']).read_bytes() != Path(args.prompt_data).read_bytes():
        raise ValueError('Resume changes training schedule')
    args.start_rollout_id = start
    return dict(iteration=iteration, start_rollout_id=start, actor_updates=start,
                question_cursor=str(cursor.resolve()), checkpoint=str(checkpoint.resolve()),
                group_size_change=group_change)


def validate_actor_restore(resume, reports, batch_size, world_size):
    expected = resume['actor_updates']
    if len(reports) != world_size:
        raise ValueError('Missing actor optimizer ranks')
    for rank, report in enumerate(reports):
        if (report['steps'] != [expected] or report['states'] <= 0
                or report['scheduler_samples'] != expected * batch_size
                or report['no_load_optim'] or report['no_load_rng'] or report['finetune']):
            raise ValueError(f'Actor rank {rank} failed optimizer restoration: {report}')
    return dict(passed=True, resume=resume, world_size=world_size, reports=reports)
