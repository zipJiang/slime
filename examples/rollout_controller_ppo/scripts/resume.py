"""Validate paired PPO checkpoints before allocating any model actors."""
import copy
import json
from pathlib import Path
from recipe import RECIPE_ID


def role_arguments(args):
    actor, critic = copy.deepcopy(args), copy.deepcopy(args)
    if args.ppo_critic_load is None:
        # Native Slime marks HF initialization as finetuning, without optimizer
        # or RNG load. A native actor checkpoint alone is not a PPO resume.
        if (Path(args.load)/'latest_checkpointed_iteration.txt').exists():
            raise ValueError('PPO resume requires both actor and critic checkpoints')
        if args.start_rollout_id not in (None, 0):
            raise ValueError('Fresh HF initialization must start at collection round zero')
        return actor, critic, None

    roots = {'actor': Path(args.load), 'critic': Path(args.ppo_critic_load)}
    for root in roots.values():
        recipe = json.loads((root.parent/'recipe.json').read_text())
        if recipe.get('recipe_id') != RECIPE_ID:
            raise ValueError('Cannot resume a different estimator recipe')
    if roots['actor'].parent.resolve() != roots['critic'].parent.resolve():
        raise ValueError('Actor and critic must come from the same run')
    if any((root.parent/'scientific-rejection.json').exists() for root in roots.values()):
        raise ValueError('A scientifically rejected run cannot be resumed for training')
    iterations = {}
    for role, root in roots.items():
        iterations[role] = int((root/'latest_checkpointed_iteration.txt').read_text().strip())
    if len(set(iterations.values())) != 1:
        raise ValueError(f'Actor/critic checkpoint iterations disagree: {iterations}')
    iteration = iterations['actor']
    start = iteration + 1
    if args.start_rollout_id is not None and args.start_rollout_id != start:
        raise ValueError('Resume cursor must follow the saved paired checkpoint')
    if args.ckpt_step is not None and args.ckpt_step != iteration:
        raise ValueError('Explicit checkpoint step disagrees with the paired trackers')
    if args.finetune:
        raise ValueError('PPO resume must not use finetune iteration reset')
    actor_updates = max(0, start - args.num_critic_only_steps)
    for role, root in roots.items():
        expected = actor_updates if role == 'actor' else start
        checkpoint = root/f'iter_{iteration:07d}'
        audit = json.loads((root/f'iter_{iteration:07d}-readback.json').read_text())
        if (Path(audit['checkpoint']).resolve() != checkpoint.resolve()
                or audit['role'] != role or audit['expected_optimizer_steps'] != expected
                or audit['optimizer_steps'] != [expected]
                or not audit['full_storage_read'] or not audit['finite_tensors']):
            raise ValueError(f'{role} checkpoint lacks a matching full readback audit')
    cursor = roots['actor']/'rollout'/f'global_dataset_state_dict_{iteration}.pt'
    if not args.rollout_global_dataset or not cursor.is_file():
        raise ValueError('PPO resume requires the saved question cursor')
    # A never-stepped actor has no Adam moments. Loading its model/RNG while
    # constructing a fresh, zero-step optimizer preserves that exact state.
    # The trained critic MUST restore its optimizer even during actor warm-up.
    actor.no_load_optim = actor_updates == 0
    critic.no_load_optim = False
    actor.no_load_rng = critic.no_load_rng = False
    actor.finetune = critic.finetune = False
    critic.load = str(roots['critic'])
    return actor, critic, dict(iteration=iteration, start_rollout_id=start,
        actor_updates=actor_updates, critic_updates=start,
        cold_actor_optimizer=actor_updates == 0, question_cursor=str(cursor.resolve()),
        checkpoints={role:str(root.resolve()) for role, root in roots.items()})
