"""Explicit collection profiles with immutable legacy and robust runtimes."""
import os
from pathlib import Path
from runtime_v2 import contract, digest, make_world as original_world
from runtime_v3 import activate as activate_robust
from runtime_v3 import contract as robust_contract, make_world as robust_world

if os.environ.get('LOC_COLLECTION_PROFILE') == 'robust-null-v3':
    activate_robust()

PROFILES = ('original', 'reply8', 'compact16-reply8')
COLLECTION_PROFILES = PROFILES + ('compact24-reply8-read60', 'robust-null-v3')


def environment(profile='original'):
    if profile not in COLLECTION_PROFILES:
        raise ValueError('Unknown collection profile: '+str(profile))
    if profile == 'compact24-reply8-read60':
        from synthetic_runtime import environment as imitation_environment
        return imitation_environment()
    if profile == 'robust-null-v3':
        return robust_contract()
    result = contract()
    if profile != 'original':
        result.update(profile='locbench-focused-coding-v2/'+profile, actor_reply_limit=8192)
        if profile == 'compact16-reply8':
            result['prompt_limit'] = 16384
    return result


def identify(value):
    for profile in COLLECTION_PROFILES:
        if value == environment(profile):
            return profile
    raise ValueError('Unknown or altered collection environment')


def validate_environment(recipe):
    profile = identify(recipe['environment'])
    if profile != 'original' and recipe.get('collection_profile_sha256') != digest(__file__):
        raise ValueError('Collection profile implementation changed')
    return profile


def make_world(case, repo, policy, *, profile='original'):
    config = environment(profile)
    if profile == 'compact24-reply8-read60':
        from synthetic_runtime import make_world as imitation_world
        return imitation_world(case, repo, policy)
    if profile == 'robust-null-v3':
        return robust_world(case, repo, policy)
    runner, workspace, prompt, tools = original_world(case, repo, policy)
    # Per-world instance: change only the trigger, keeping the tested fold prompt,
    # token allowance, completion checks, budget refill and global horizon.
    if config['prompt_limit'] != contract()['prompt_limit']:
        from step_controller.harness.compaction.triggers import AnyOf, PromptTokens, StateCounter
        runner._compactor.trigger = AnyOf(PromptTokens(config['prompt_limit']), StateCounter())
    return runner, workspace, prompt, tools


def transition(previous, requested, *, allow=False):
    old = identify(previous)
    current = environment(requested)
    if current == previous:
        return None
    supported = (
        old == 'original' and requested != 'original'
    ) or (
        old == 'compact24-reply8-read60' and requested == 'robust-null-v3'
    )
    if not allow or not supported:
        raise ValueError('Collection profile change requires an explicit supported transition')
    return dict(previous=previous, current=current,
                scope='Fresh collections only; paired weights, optimizer, cursor and critic context preserved',
                critic_transfer='Critic weights transfer explicitly; on-policy updates recalibrate the changed serialized context',
                efficiency_transfer='Prior topology and search budget adopted; new profile efficiency not benchmark-equivalent')
