"""Explicit collection profiles; frozen critic serialization remains runtime_v2."""
from pathlib import Path
from runtime_v2 import contract, digest, make_world as original_world

PROFILES = ('original', 'reply8', 'compact16-reply8')
COLLECTION_PROFILES = PROFILES + ('compact24-reply8-read60',)


def environment(profile='original'):
    if profile not in COLLECTION_PROFILES:
        raise ValueError('Unknown collection profile: '+str(profile))
    if profile == 'compact24-reply8-read60':
        from synthetic_runtime import environment as imitation_environment
        return imitation_environment()
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
    if not allow or old != 'original' or requested == 'original':
        raise ValueError('Collection profile change requires an explicit supported transition')
    return dict(previous=previous, current=current,
                scope='Fresh collections only; paired weights, optimizer, cursor and critic context preserved',
                critic_transfer='Critic learned under the original collection profile',
                efficiency_transfer='Prior topology and search budget adopted; new profile efficiency not benchmark-equivalent')
